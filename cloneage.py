import sys
from math import comb

import vptree, tlsh
import pathlib, os
import polars as pl
import pandas as pd

import cudf, cugraph, gc, rmm
import tqdm
import json


# INITIALIZE MANAGED MEMORY
rmm.reinitialize(
    managed_memory=True,     # Allows "swapping" to system RAM
    initial_pool_size=None,  # Lets it grow as needed
)


# ==============================
# LIGHTWEIGHT OBJECT
# ==============================

class Lib:
    __slots__ = ("id", "key", "size", "processed", "tlsh", "idf")

    def __init__(self, id, key, size, processed, tlsh, idf):
        self.id = id
        self.key = key
        self.size = size
        self.processed = processed
        self.tlsh = tlsh
        self.idf = idf


# ==============================
# PARQUET WRITER
# ==============================

class ParquetEdgeWriter:
    def __init__(self, output_dir=pathlib.Path("datasets/edges")):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.file_count = self._get_last_index()

    def _get_last_index(self):
        files = list(self.output_dir.glob("part_*.parquet"))

        if not files:
            return 0

        indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files]

        return max(indices) + 1 if indices else 0

    def write_batch(self, batch):
        if not batch:
            return

        try:
            df = pd.DataFrame(batch, columns=["from", "to", "distance", "method", "strategy", "tversky_score"])
            filename = os.path.join(self.output_dir, f"part_{self.file_count:06d}.parquet")
            df.to_parquet(filename, index=False, compression="zstd")
            self.file_count += 1

        except Exception as e:
            print(f"[ERROR] Failed writing batch: {e}")
            try:
                fallback = os.path.join(self.output_dir, f"fallback_{self.file_count:06d}.csv")
                df.to_csv(fallback, index=False)
            except Exception as e:
                print(f"[ERROR] Failed writing backup batch: {e}")


# ==============================
# SIMILARITY ENGINE
# ==============================
class SimilarityEngine:
    def __init__(self, output_dir=pathlib.Path("datasets/edges")):
        self.batch_size = 10000000
        self.threshold_radius = 50  # tau_{radius}
        self.count = 0
        self.writer = ParquetEdgeWriter(output_dir=output_dir)

    def _save_batch(self, batch):
        if batch:
            self.count += len(batch)
            self.writer.write_batch(batch)
        return []

    def _create_edge(self, a, b, dist, mode):
        # canonical ordering ALWAYS
        from_id = min(a.id, b.id)
        to_id = max(a.id, b.id)

        return (from_id, to_id, int(dist), "TLSH", mode, round(max(0, 1 - (dist / 100)), 4))

    def _update_processed_status(self, processed_keys):
        if not processed_keys:
            return

        print("[+] Persisting processed status to CSV...")
        df = pd.read_csv('datasets/libraries.csv')

        df.loc[df['key'].isin(processed_keys), 'processed'] = True

        df.to_csv('datasets/libraries.csv', index=False)

        print("[+] Status persisted.")

    # ==============================
    # ANN (VP-Tree)
    # ==============================

    def run_ann(self, libs):
        print(f"[!] Building VP-Tree with {len(libs)} libraries...")

        def dist_func(l1, l2):
            return tlsh.diff(l1.tlsh, l2.tlsh)

        idf_values = pl.Series([l.idf for l in libs])
        threshold_idf = idf_values.quantile(0.99)
        libs = [l for l in libs if l.idf >= threshold_idf]
        new_libs = [l for l in libs if not l.processed]

        if not new_libs:
            print("[+] No new libraries to process.")
            return

        print(f"[+] Processing {len(new_libs)} new libraries...")
        tree = vptree.VPTree(libs, dist_func)

        batch = []
        processed_keys = []

        try:
            for lib in tqdm(new_libs, desc="ANN Incremental"):

                neighbors = tree.within(lib, self.threshold_radius)

                for neighbor, dist in neighbors:

                    # A. avoid self and B. GLOBAL CANONICAL ORDER (REAL FIX)
                    if neighbor.id == lib.id or neighbor.id <= lib.id:
                        continue

                    # C. size filter
                    if abs(lib.size - neighbor.size) > lib.size * 0.25:
                        continue

                    batch.append(self._create_edge(lib, neighbor, dist, "ann_vptree"))

                    if len(batch) >= self.batch_size:
                        batch = self._save_batch(batch)

                lib.processed = True
                processed_keys.append(lib.key)

            if len(batch):
                self._save_batch(batch)

            if processed_keys:
                self._update_processed_status(processed_keys)

        except KeyboardInterrupt:
            print(f"\n[!] Interrupt detected. Saving progress ({len(processed_keys)} libraries)...")
            if batch:
                self._save_batch(batch)
            if processed_keys:
                self._update_processed_status(processed_keys)
            print("[+] Progress saved. Exiting safely.")
            import sys
            sys.exit(0)


def fetch_libraries(path_libs, path_mapping='datasets/id_mapping.csv'):
    # 1. Library Loading
    if not os.path.exists(path_libs):
        sys.exit('[x] You need to dump libraries collection before you continue!!!')

    df_raw = pl.read_csv(path_libs)

    # 2. Mapping and Validations
    if os.path.exists(path_mapping):
        mapping = pl.read_csv(path_mapping)

        if mapping["key"].is_duplicated().any():
            raise ValueError("[FATAL] Duplicate keys in mapping")
        if mapping["id"].is_duplicated().any():
            raise ValueError("[FATAL] Duplicate IDs in mapping")
    else:
        sys.exit(f'[x] Mapping file {path_mapping} not found!')

    # 3. Merge (Join in Polars)
    df_final = df_raw.join(mapping, on="key", how="left")

    # Null validation after join
    if df_final["id"].null_count() > 0:
        raise ValueError("[FATAL] Missing IDs after merge")

    # 4. Lib object construction
    print(f"[+] Processing {len(df_final)} libraries...")

    columnas = ["id", "key", "size", "processed", "tlsh", "idf"]

    libs = [
        Lib(*row)
        for row in df_final.select(columnas).iter_rows()
    ]

    return libs

# --- STEP 1: IDF Weighting ---

def calculate_idf_libraries(libraries_path, extlib_path, total_exts_unique):
    libs_df = pl.read_csv(libraries_path)
    extlib_df = pl.scan_csv(extlib_path)

    usage_counts = (
        extlib_df
        .group_by("lib_id")
        .agg(pl.col("ext_id").n_unique().alias("usage"))
        .collect(engine="streaming")
    )

    libs_df = (
        libs_df
        .drop(["usage"], strict=False)
        .join(usage_counts, left_on="key", right_on="lib_id", how="left")
        .with_columns(
            usage=pl.col("usage").fill_null(0)
        )
        .with_columns(
            idf=pl.when(pl.col("usage") > 0)
            .then((total_exts_unique / pl.col("usage")).log10())
            .otherwise(0.0)
        )
    )

    libs_df.write_csv(libraries_path)

    print(f"[✓] IDF Updated. IDF stats: \n {libs_df['idf'].describe()}")
    return libs_df


# --- STEP 2: Fuzzy Library Clustering (Louvain) ---
def run_louvain_clustering(edge_pattern, threshold):
    q = pl.scan_parquet(edge_pattern)

    print(f"[+] Filtering structural skeleton (dist <= {threshold})...")
    skeleton_edges = (
        q.filter(pl.col("distance") <= threshold)
        .select([
            pl.col("from").cast(pl.Int32),
            pl.col("to").cast(pl.Int32),
            pl.col("distance").cast(pl.Float32)
        ])
        .collect(engine="streaming")
    )

    print(f"[!] Resulting edges: {len(skeleton_edges)}")
    # 1. Using GPU
    gdf = cudf.from_pandas(skeleton_edges.to_pandas())

    # 2. Build Graph (Affinity = Threshold - Distance)
    gdf["weight"] = (threshold + 1) - gdf["distance"]
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source='from', destination='to', edge_attr='weight')

    del gdf
    gc.collect()

    # 3. Louvain Community Detection
    df_clusters, modularity = cugraph.louvain(G)

    # 4. Back to CPU for final mapping
    vertex_to_cluster = df_clusters.to_pandas()
    # Final cleanup
    del G
    del df_clusters
    gc.collect()

    return vertex_to_cluster, modularity


# --- STEP 3: Structural Rarity Weighting ---
def calculate_final_cluster_weights(cluster_mapping, extlib_path, mapping_path, total_exts_unique):
    print("[+] Calculating final weights per cluster (df_c) using unique ext_id...")

    # 1. Clusters (numeric ID -> Cluster ID)
    clusters = (
        pl.from_pandas(cluster_mapping)
        .rename({"vertex": "id_num", "partition": "cluster_id"})
        .select([
            pl.col("id_num").cast(pl.Int64),
            pl.col("cluster_id").cast(pl.Int32)
        ])
        .lazy()
    )

    # 2. SHA -> numeric ID mapping (necessary bridge for Louvain)
    mapping = (
        pl.scan_csv(mapping_path)
        .select([
            pl.col("key").alias("sha_key"),
            pl.col("id").cast(pl.Int64).alias("id_num")
        ])
    )

    # 3. extlib (ext_id is already the unique extension ID)
    extlib_lazy = (
        pl.scan_csv(extlib_path)
        .select(["ext_id", "lib_id"])
        .rename({"lib_id": "sha_key"})
    )

    # 4. JOIN & AGGREGATION
    df_c = (
        extlib_lazy
        .join(mapping, on="sha_key")
        .join(clusters, on="id_num")
        .select(["ext_id", "cluster_id"])
        .unique()  # ---> One cluster per extension (unique entity)
        .group_by("cluster_id")
        .agg(pl.len().alias("count"))  # ---> len() after unique() counts unique extensions
        .collect(engine='streaming')
    )

    # 5. Structural IDF
    weights = df_c.with_columns(
        w_c=(total_exts_unique / pl.col("count")).log10()
    )

    return weights


def generate_final_library_weights(vertex_to_cluster, cluster_weights, libraries_path, mapping_path):
    print("[+] Consolidating final weights and library metadata...")

    # 1. Load libraries (Lazy)
    libs_lazy = pl.scan_csv(libraries_path)

    # 2. Load mapping (Lazy) to get the numeric ID
    mapping_lazy = pl.scan_csv(mapping_path).select([
        pl.col("key"),
        pl.col("id").cast(pl.Int64)
    ])

    # 3. Prepare clusters (from Pandas/GPU DataFrame)
    clusters_lazy = (
        pl.from_pandas(vertex_to_cluster)
        .rename({"vertex": "id", "partition": "cluster_id"})
        .select([
            pl.col("id").cast(pl.Int64),
            pl.col("cluster_id").cast(pl.Int32)
        ])
        .lazy()
    )

    # 4. Cluster weights (Lazy)
    weights_lazy = cluster_weights.lazy()

    # 5. Final Join: Metadata + IDs + Clusters + Weights
    final_libs = (
        libs_lazy
        # Join metadata with numeric ID using SHA (key)
        .join(mapping_lazy, on="key", how="left")
        # Join with clusters using numeric ID
        .join(clusters_lazy, on="id", how="left")
        # Join with calculated weights per cluster
        .join(weights_lazy, on="cluster_id", how="left")
        .with_columns(
            # If the library belongs to a cluster, use the cluster weight (w_c).
            # If isolated (w_c is null), keep its original IDF.
            final_weight=pl.col("w_c").fill_null(pl.col("idf"))
        )
        .collect(engine='streaming')
    )

    return final_libs


def compute_extension_similarity(ext_to_clusters, cluster_weights, threshold=0.6, min_wc=0.0):
    print(f"[+] Filtering clusters with low information gain (w_c < {min_wc})...")

    # 1. Fetch weights and filter by INFORMATION VALUE
    weights = (
        cluster_weights
        .select(["cluster_id", "w_c", "count"])
        .filter(pl.col("w_c") >= min_wc)  # <--- The magic happens here
        .lazy()
    )

    # 2. Join with profiles
    df = ext_to_clusters.lazy().join(weights, on="cluster_id")

    # 3. Calculate W_X (Totals per extension)
    # Important: W_X must be calculated AFTER filtering common clusters
    # so that similarity is based only on "distinctive" code.
    ext_totals = (
        df.group_by("extension_id")
        .agg(pl.col("w_c").sum().alias("total_weight"))
    )

    # 4. Self-join (now much safer because we removed low-IDF clusters)
    lean_df = df.select(["extension_id", "cluster_id", "w_c"])

    pairs = (
        lean_df.join(lean_df, on="cluster_id", suffix="_right")
        .filter(pl.col("extension_id") < pl.col("extension_id_right"))
    )

    # 6. Sum intersections W_M
    intersections = (
        pairs.group_by(["extension_id", "extension_id_right"])
        .agg(pl.col("w_c").sum().alias("W_M"))
    )

    # 7. Join with totals and calculate Tversky
    final_scores = (
        intersections
        .join(ext_totals, on="extension_id")
        .join(ext_totals, left_on="extension_id_right", right_on="extension_id", suffix="_B")
        .with_columns([
            pl.col("total_weight").alias("W_A"),
            pl.col("total_weight_B").alias("W_B")
        ])
        .with_columns(
            alpha=pl.col("W_A") / (pl.col("W_A") + pl.col("W_B")),
            beta=pl.col("W_B") / (pl.col("W_A") + pl.col("W_B"))
        )
        .with_columns(
            W_A_only=(pl.col("W_A") - pl.col("W_M")).clip(lower_bound=0),
            W_B_only=(pl.col("W_B") - pl.col("W_M")).clip(lower_bound=0)
        )
        .with_columns(
            score=pl.col("W_M") / (
                        pl.col("W_M") + (pl.col("alpha") * pl.col("W_A_only")) + (pl.col("beta") * pl.col("W_B_only")))
        )
        .filter(pl.col("score") >= threshold)
    )

    return final_scores.collect(engine='streaming')


def generate_extension_profiles(extlib_path, final_libs_df):
    print("[+] Generating cluster profiles per extension (C_X)...")

    # 1. Select only the necessary mapping from our already-processed libraries
    # Using final_weight because it already consolidates cluster vs original idf
    lib_mapping = final_libs_df.select([
        pl.col("key").alias("lib_sha"),
        pl.col("cluster_id"),
        pl.col("final_weight")
    ]).lazy()

    # 2. Load extension-library relationships
    ext_profiles = (
        pl.scan_csv(extlib_path)
        .select([
            pl.col("ext_id").alias("extension_id"),
            pl.col("lib_id").alias("lib_sha")
        ])
        # Join to find which cluster and weight each extension file belongs to
        .join(lib_mapping, on="lib_sha")
        # --- KEY STEP ---
        # Group by extension and cluster so each cluster counts ONLY ONCE (Set of clusters)
        .group_by(["extension_id", "cluster_id"])
        .agg(pl.max("final_weight").alias("final_weight"))
        .collect(engine='streaming')
    )
    return ext_profiles


def _build_ext_metadata(extensions_path: str) -> pl.DataFrame:

    json_schema = pl.Struct([pl.Field("name", pl.String)])

    raw = (
        pl.scan_csv(extensions_path)
        .with_columns(
            pl.col("obsoleteReason").fill_null("safe"),
            pl.col("alive").cast(pl.Boolean),
            pl.col("in_cws").cast(pl.Boolean),
            pl.col("downloadable").cast(pl.Boolean),
        )
    )

    # -- Malware aggregation (per id+version first, then per id) -- #
    per_version = (
        raw
        .select(["id", "version", "obsoleteReason"])
        .group_by(["id", "version"])
        .agg(pl.col("obsoleteReason").first().alias("reason"))
    )

    malware_agg = (
        per_version
        .group_by("id")
        .agg([
            (pl.col("reason") != "safe").any().alias("malware"),
            pl.col("version").filter(pl.col("reason") != "safe").alias("versions"),
            pl.col("reason").filter(pl.col("reason") != "safe").alias("reasons"),
        ])
        .with_columns(
            pl.struct(["versions", "reasons"])
            .map_elements(
                lambda s: (
                    "safe" if len(s["versions"]) == 0
                    else json.dumps(dict(zip(s["versions"], s["reasons"])))
                ),
                return_dtype=pl.String,
            )
            .alias("malware_reason")
        )
        .select(["id", "malware", "malware_reason"])
    )

    # -- Alive aggregation -- #
    alive_agg = (
        raw
        .select(["id", "alive", "in_cws", "downloadable"])
        .group_by("id")
        .agg([
            pl.col("alive").any(),
            pl.col("in_cws").any(),
            pl.col("downloadable").any(),
        ])
    )

    # -- Name -- #
    name_agg = (
        raw
        .select([
            pl.col("id"),
            pl.col("metadata")
              .str.json_decode(dtype=json_schema)
              .struct.field("name")
              .fill_null("unknown")
              .alias("name"),
        ])
        .group_by("id")
        .agg(pl.col("name").first())
    )

    # -- Combine all -- #
    meta = (
        malware_agg
        .join(alive_agg,  on="id", how="left")
        .join(name_agg,   on="id", how="left")
        .collect()
    )

    return meta


def enrich_lineages_with_metadata(lineages_path, extensions_path):
    print("[+] Enriching lineages with metadata...")

    meta = _build_ext_metadata(extensions_path)

    # Side A
    meta_A = meta.rename({
        "id":             "extension_id",
        "malware":        "malware_A",
        "malware_reason": "malware_A_reason",
        "alive":          "alive_A",
        "in_cws":         "in_cws_A",
        "downloadable":   "downloadable_A",
        "name":           "name_A",
    })

    # Side B
    meta_B = meta.rename({
        "id":             "extension_id_right",
        "malware":        "malware_B",
        "malware_reason": "malware_B_reason",
        "alive":          "alive_B",
        "in_cws":         "in_cws_B",
        "downloadable":   "downloadable_B",
        "name":           "name_B",
    })

    final_df = (
        pl.scan_parquet(lineages_path)
        .join(meta_A.lazy(), on="extension_id",       how="left")
        .join(meta_B.lazy(), on="extension_id_right", how="left")
        .select([
            "name_A", "name_B",
            "malware_A", "malware_A_reason",
            "malware_B", "malware_B_reason",
            "alive_A", "in_cws_A", "downloadable_A",
            "alive_B", "in_cws_B", "downloadable_B",
            "score", "W_M", "W_A", "W_B",
            "extension_id", "extension_id_right",
        ])
        .sort("score", descending=True)
        .unique(subset=["extension_id", "extension_id_right"], keep="first")
        .sort(["extension_id", "extension_id_right", "score"], descending=False)
        .collect(engine="streaming")
    )

    return final_df


def detect_extension_families(extension_lineages):
    import networkx as nx
    from networkx.algorithms import community

    print("[+] Building family graph and detecting cohesive communities...")

    # 1. Create the graph
    G = nx.Graph()

    # 2. Feed the graph INCLUDING score as weight
    # By passing the score, the algorithm knows which links are strong and which are weak bridges
    edges = extension_lineages.select([
        "extension_id", "extension_id_right", "score"
    ]).iter_rows()

    for src, dst, score in edges:
        G.add_edge(src, dst, weight=float(score))

    # 3. EXTRACT COMMUNITIES
    # Louvain will cut weak threads and keep dense blocks separated
    communities = community.louvain_communities(G, weight="weight")
    components = [list(c) for c in communities]

    # 4. Create a mapping DataFrame: [extension_id, family_id, family_size]
    family_data = []
    for i, component in enumerate(components):
        size = len(component)
        for ext_id in component:
            family_data.append((ext_id, i, size))

    df_families = pl.DataFrame(
        family_data,
        schema=["extension_id", "family_id", "family_size"]
    )

    print(f"[✓] {len(components)} unique real families detected.")
    return df_families, components


def generate_simplified_family_report(df_families, extensions_path):
    print("[+] Generating family report with full metadata...")

    meta = (
        _build_ext_metadata(extensions_path)
        .rename({
            "id":      "extension_id",
            "malware": "malware_A",
        })
        .unique(subset=["extension_id"], keep="first")
    )

    return df_families.join(meta, on="extension_id", how="left")


def generate_gephi_nodes(df_families, extensions_path):
    print("[+] Generating Gephi node table...")

    meta = (
        _build_ext_metadata(extensions_path)
        .rename({
            "id":             "Id",
            "malware":        "Malware_Status",
            "malware_reason": "Malware_Reason",
            "name":           "Label",
        })
        .unique(subset=["Id"], keep="first")
    )

    nodes = (
        df_families
        .rename({"extension_id": "Id"})
        .join(meta, on="Id", how="left")
        .with_columns(
            pl.when(pl.col("Malware_Status"))
              .then(pl.lit("Red"))
              .otherwise(pl.lit("Green"))
              .alias("Color")
        )

        .select([
            "Id", "family_id", "family_size",
            "Malware_Status", "Malware_Reason",
            "Label", "Color",
            "alive", "in_cws", "downloadable",
        ])
    )

    return nodes


def generate_gephi_edges(extension_lineages):
    print("[+] Generating Gephi edge table...")

    edges = (
        extension_lineages
        .select([
            pl.col("extension_id").alias("Source"),
            pl.col("extension_id_right").alias("Target"),
            pl.col("score").alias("Weight"),  # Tversky score is the weight
            pl.lit("Undirected").alias("Type")
        ])
    )
    return edges


if __name__ == "__main__":
    '''
    Steps 1, 2, 3, 4, 5 are for the:
        1) Identity-based Pruning and Pre-processing
        2) Fuzzy Library Clustering
        3) Cluster-level Structural Weighting
    '''

    # --- Configuration ---
    DATASETS_PATH = pathlib.Path("datasets")
    RESULTS_PATH = pathlib.Path("results")
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    DATASETS_PATH.mkdir(parents=True, exist_ok=True)

    EDGE_DIR = DATASETS_PATH / 'edges'
    MAPPING_PATH = DATASETS_PATH / "id_mapping.csv"
    LIBRARIES_PATH = DATASETS_PATH / "libraries.csv"
    EXTENSIONS_PATH = DATASETS_PATH / "extensions.csv"
    EXTLIB_PATH = DATASETS_PATH / "ext_lib.csv"

    taus = [15, 25, 35]
    scores = [0.95, 0.8, 0.6, 0.4]

    N = (
        pl.scan_csv(EXTENSIONS_PATH)
        .select("id")
        .unique()
        .collect()
        .height
    )

    # # 1. Identity-based Pruning

    # calculate_idf_libraries(LIBRARIES_PATH, EXTLIB_PATH, total_exts_unique=N)

    # # 2. Similarity Search (VP-Tree)
    # # This is a computationally demanding operation!!! BE AWARE!!!

    # engine = SimilarityEngine(output_dir=EDGE_DIR)
    # libs = fetch_libraries(LIBRARIES_PATH)
    # engine.run_ann(libs)

    # 3. Clustering (Louvain on GPU)
    threshold_skeleton = 8
    vertex_to_cluster, modularity = run_louvain_clustering(EDGE_DIR / "part_*.parquet", threshold=threshold_skeleton)

    # 4. Cluster Rarity Weighting
    # df(c) = |\{X : H_X \cap c \neq \emptyset\}|
    cluster_weights = calculate_final_cluster_weights(vertex_to_cluster, EXTLIB_PATH, MAPPING_PATH, N)

    # 5. Final Weights
    final_libs = generate_final_library_weights(vertex_to_cluster, cluster_weights, LIBRARIES_PATH, MAPPING_PATH)
    final_libs.write_parquet(RESULTS_PATH / f"libraries_with_clusters_skel_{threshold_skeleton}.parquet")
    cluster_weights.write_parquet(RESULTS_PATH / f"cluster_weights_skel_{threshold_skeleton}.parquet")

    # 6. Tversky Index
    ext_profiles = generate_extension_profiles(EXTLIB_PATH, final_libs)
    output_profiles = RESULTS_PATH / f"extension_profiles_skel_{threshold_skeleton}.parquet"
    ext_profiles.write_parquet(output_profiles)
    print(f"[✓] Profiles stored: {output_profiles}")

    # 7. Lineages (Relations X -> Y)
    tau = 0.9
    extension_lineages = compute_extension_similarity(ext_profiles.lazy(), cluster_weights.lazy(), threshold=tau)
    output_lineages = RESULTS_PATH / f"extension_lineages_skel_{threshold_skeleton}_tau_{tau}.parquet"
    extension_lineages.write_parquet(output_lineages)

    # 8. Adding metadata to lineages (Relations X -> Y)
    final_df = enrich_lineages_with_metadata(output_lineages, EXTENSIONS_PATH)
    output_path = RESULTS_PATH / f"extension_lineages_metadata_skel_{threshold_skeleton}_tau_{tau}.parquet"
    final_df.write_csv(output_path.with_suffix(".csv"))
    final_df.write_parquet(output_path.with_suffix(".parquet"))

    # 9. Family analysis
    df_families, families_list = detect_extension_families(extension_lineages)
    final_report = generate_simplified_family_report(df_families, EXTENSIONS_PATH)
    output_families = RESULTS_PATH / f"families_detected_skel_{threshold_skeleton}_tau_{tau}.csv"
    final_report.sort("family_size", descending=True).write_csv(output_families)

    exts_con_familia = df_families["extension_id"].n_unique()
    print(f"Total extensions: {N}")
    print(f"Extensions with relatives: {exts_con_familia}")
    print(f"Orphan extensions (unique): {N - exts_con_familia}")

    # 10. Gephi preparation
    nodes_df = generate_gephi_nodes(df_families, EXTENSIONS_PATH)
    edges_df = generate_gephi_edges(extension_lineages)
    nodes_df.write_csv(RESULTS_PATH / f"gephi_nodes_skel_{threshold_skeleton}_tau_{tau}.csv")
    edges_df.write_csv(RESULTS_PATH / f"gephi_edges_skel_{threshold_skeleton}_tau_{tau}.csv")

    print(f"[✓] Pipeline DONE.")