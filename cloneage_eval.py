import polars as pl
import pandas as pd
import pathlib
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import spearmanr
import igraph as ig



# Execution

DATASETS_PATH = pathlib.Path("datasets")
RESULTS_PATH = pathlib.Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
DATASETS_PATH.mkdir(parents=True, exist_ok=True)

EDGE_DIR = DATASETS_PATH / 'edges'
MAPPING_PATH = DATASETS_PATH / "id_mapping.csv"
LIBRARIES_PATH = DATASETS_PATH / "libraries.csv"
EXTENSIONS_PATH = DATASETS_PATH / "extensions.csv"
EXTLIB_PATH = DATASETS_PATH / "ext_lib.csv"




def clustering_evaluation():
    from cloneage import run_louvain_clustering

    threshold_skeletons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    results_all = []

    q = pl.scan_parquet(EDGE_DIR / "part_*.parquet")

    # =========================================================
    # SPID FUNCTION
    # =========================================================
    import numpy as np
    import cupy as cp
    import cudf
    import cugraph

    def compute_metrics(edges_df, nodes_gpu):

        nodes_np = nodes_gpu.to_numpy()
        N_sub = len(nodes_np)

        if N_sub < 2:
            return None, None, None, None, None

        sample_size = min(100, N_sub)
        sample_nodes = np.random.choice(nodes_np, size=sample_size, replace=False)

        nodes_df = cudf.DataFrame({"vertex": nodes_gpu})

        sub_edges = edges_df.merge(nodes_df, left_on="from", right_on="vertex") \
            .merge(nodes_df, left_on="to", right_on="vertex")

        if len(sub_edges) == 0:
            return None, None, None, None, None

        G_sub = cugraph.Graph()
        G_sub.from_cudf_edgelist(sub_edges, "from", "to", "weight")

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0
        max_dist = 0.0

        total_expected = sample_size * (N_sub - 1)
        actual_reachable = 0

        for src in sample_nodes:
            bfs_df = cugraph.bfs(G_sub, start=int(src))
            d = bfs_df["distance"]

            # Filter distance 0 (source) and infinity (unreachable nodes)
            d_valid = d[(d > 0) & (d < 2147483647)]
            actual_reachable += len(d_valid)

            if len(d_valid) == 0:
                continue

            d_cp = cp.asarray(d_valid)

            total_sum += float(cp.sum(d_cp))
            total_sq_sum += float(cp.sum(d_cp ** 2))
            total_count += len(d_cp)

            max_dist = max(max_dist, float(cp.max(d_cp)))

        if total_count == 0:
            return None, None, None, None, None

        mean = total_sum / total_count
        var = (total_sq_sum / total_count) - mean ** 2

        sampled_spid = var / mean if mean > 0 else 0.0
        reachability = actual_reachable / total_expected if total_expected > 0 else 0.0

        adjusted_spid = sampled_spid / reachability if reachability > 0 else sampled_spid

        return sampled_spid, adjusted_spid, mean, reachability, max_dist

    # =========================================================
    # MAIN LOOP
    # =========================================================
    for ts in tqdm(threshold_skeletons, desc="Threshold sweep"):

        # CONSISTENT EDGE SET
        edges_df = (
            q.filter(pl.col("distance") <= ts)
            .select([
                pl.col("from").cast(pl.Int32),
                pl.col("to").cast(pl.Int32),
                pl.col("distance").cast(pl.Float32)
            ])
            .collect(engine="streaming")
        )

        edges_df = cudf.from_pandas(edges_df.to_pandas())
        edges_df["weight"] = (ts + 1) - edges_df["distance"]

        # -------------------------
        # clustering
        # -------------------------
        df_clusters_gpu, modularity = run_louvain_clustering(
            EDGE_DIR / "part_*.parquet",
            threshold=ts
        )

        cluster_sizes = (
            df_clusters_gpu
            .groupby("partition")
            .size()
            .reset_index(name="size")
        )

        def assign_bin(size):
            if size < 10:
                return "tiny"
            elif size < 50:
                return "small"
            elif size < 200:
                return "medium"
            elif size < 1000:
                return "large"
            else:
                return "xlarge"

        cluster_sizes["bin"] = cluster_sizes["size"].apply(assign_bin)

        sampled_clusters = (
            cluster_sizes.groupby("bin", group_keys=True)
            .apply(lambda x: x.sample(min(len(x), 500)), include_groups=False)
            .reset_index(level=0)
            .drop_duplicates(subset=["partition"])
        )

        cluster_ids = sampled_clusters["partition"].tolist()

        coverage_stats = (cluster_sizes.groupby("bin").size().reset_index(name="total_clusters"))
        sampled_stats = (sampled_clusters.groupby("bin").size().reset_index(name="sampled_clusters"))
        coverage = coverage_stats.merge(sampled_stats, on="bin")
        coverage["coverage_pct"] = (100 * coverage["sampled_clusters"] / coverage["total_clusters"])

        coverage_map = (
            coverage
            .set_index("bin")
            .to_dict(orient="index")
        )
        # =========================================================
        # CLUSTER LOOP
        # =========================================================

        partition_to_bin = dict(zip(sampled_clusters["partition"], sampled_clusters["bin"]))

        for part_id in tqdm(cluster_ids, desc=f"SPID τ={ts}"):

            nodes = df_clusters_gpu[df_clusters_gpu["partition"] == part_id]["vertex"]

            if len(nodes) < 3:
                continue

            sampled_spid, adjusted_spid, mean_dist, reachability, diameter = compute_metrics(edges_df, nodes)

            if sampled_spid is None:
                continue

            cluster_bin = partition_to_bin[part_id]
            cov = coverage_map[cluster_bin]

            results_all.append({
                "tau_skeleton": ts,
                "size_bin": cluster_bin,
                "cluster_id": part_id,
                "size": len(nodes),

                "spid_sampled": sampled_spid,
                "spid_adjusted": adjusted_spid,
                "reachability_ratio": reachability,

                "mean_distance": mean_dist,
                "approx_diameter": diameter,
                "modularity": modularity,

                "total_clusters_bin": cov["total_clusters"],
                "sampled_clusters_bin": cov["sampled_clusters"],
                "coverage_pct_bin": cov["coverage_pct"]
            })

        del df_clusters_gpu
        cp.get_default_memory_pool().free_all_blocks()

    df = pd.DataFrame(results_all)
    df.to_csv("results/spid_cluster_sweep.csv", index=False)

    return df


def first_experiment():
    q = pl.scan_parquet(EDGE_DIR / "part_*.parquet")

    stats = (
        q.select("distance")
        .with_columns(
            bin_start=(pl.col("distance") // 10) * 10
        )
        .group_by("bin_start")
        .agg([
            pl.len().alias("n"),
            pl.col("distance").mean().alias("mu"),
            pl.col("distance").var().alias("var")
        ])
        .sort("bin_start")

        .with_columns([
            pl.col("n").sum().over(pl.lit(True)).alias("total_n"),
            pl.col("n").cum_sum().alias("cum_n"),
            (pl.col("mu") * pl.col("n")).cum_sum().alias("cum_sum"),
            ((pl.col("var") + pl.col("mu")**2) * pl.col("n")).cum_sum().alias("cum_sq")
        ])

        .with_columns([
            (pl.col("cum_sum") / pl.col("cum_n")).alias("cum_mu"),
            ((pl.col("cum_sq") / pl.col("cum_n")) - (pl.col("cum_sum") / pl.col("cum_n"))**2).alias("cum_var")
        ])

        .with_columns([
            pl.concat_str([
                pl.col("bin_start").cast(pl.String),
                pl.lit("--"),
                (pl.col("bin_start") + 10).cast(pl.String)
            ]).alias("bin"),

            (pl.col("var") / pl.col("mu")).alias("fano"),
            (pl.col("cum_var") / pl.col("cum_mu")).alias("true_acc_fano"),
            (pl.col("n") / pl.col("total_n")).alias("pct_edges")
        ])

        .select([
            "bin",
            "n",
            "mu",
            "var",
            "fano",
            "true_acc_fano",
            "pct_edges"
        ])
        .collect(engine="streaming")
    )

    print(stats)


def second_experiment():  # Fuzzy Library Clustering
    from cloneage import run_louvain_clustering
    results_log = []
    threshold_skeletons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

    for ts in threshold_skeletons:
        print(f"\n[>] Evaluating tau_skeleton = {ts}")

        # 1. Run Louvain
        # vertex_to_cluster must return (df_clusters, modularity_score)
        df_clusters, mod_score = run_louvain_clustering(EDGE_DIR / "part_*.parquet", threshold=ts)

        # 2. Extract key metrics
        num_clusters = df_clusters['partition'].nunique()
        cluster_sizes = df_clusters['partition'].value_counts()
        max_size = cluster_sizes.max()
        mean_size = cluster_sizes.mean()

        # 3. Store for analysis
        results_log.append({
            "tau_skeleton": ts,
            "modularity": mod_score,
            "n_clusters": num_clusters,
            "max_cluster_size": max_size,
            "avg_cluster_size": mean_size
        })

    df_sensitivity = pd.DataFrame(results_log)
    print(df_sensitivity)


def third_experiment():

    import polars as pl
    import pandas as pd
    from pathlib import Path

    def generate_cluster_stats(vertex_to_cluster_df, N_total):


        # 1. MAPPING: vertex → lib_id → cluster_id

        mapping = (
            pl.scan_csv(MAPPING_PATH)
            .select([
                pl.col("id").alias("vertex"),
                pl.col("key").alias("lib_id")
            ])
        )

        clusters = (
            pl.from_pandas(vertex_to_cluster_df).lazy()
            .select([
                pl.col("vertex"),
                pl.col("partition").alias("cluster_id")
            ])
            .join(mapping, on="vertex", how="left")
            .select(["lib_id", "cluster_id"])
        )


        # 2. EXTENSION-LIBRARY TABLE

        ext_libs = (
            pl.scan_csv(EXTLIB_PATH)
            .select(["lib_id", "ext_id", "version", "filename"])
            .unique()
        )


        # 3. BASE JOIN

        base = ext_libs.join(clusters, on="lib_id")


        # 4. CLUSTER STRUCTURAL STATISTICS

        cluster_stats = (
            base
            .group_by("cluster_id")
            .agg([
                pl.col("lib_id").n_unique().alias("cluster_size"),
                pl.col("ext_id").n_unique().alias("n_ext_id"),
                pl.struct(["ext_id", "version"]).n_unique().alias("n_versions"),
            ])
            .with_columns([
                (N_total / pl.col("n_ext_id")).log(10).alias("w_c"),
                (pl.col("cluster_size") / pl.col("n_ext_id")).alias("poly_ratio"),
                (pl.col("n_versions") / pl.col("n_ext_id")).alias("version_ratio"),
            ])
        )

        # 5. NAIVE FILE-LEVEL WEIGHTING (BASELINE)
        
        naive_df = (
            ext_libs
            .group_by("lib_id")
            .agg([
                pl.len().alias("raw_frequency")
            ])
            .with_columns([
                (pl.lit(N_total) / pl.col("raw_frequency")).log(10).alias("w_h")
            ])
        )


        # 6. CLUSTER-LEVEL COMPARISON (CORRECT VERSION)

        comparison_df = (
            base
            .join(naive_df, on="lib_id", how="left")
            .group_by("cluster_id")
            .agg([
                pl.col("w_h").mean().alias("w_naive_mean"),
                pl.col("w_h").median().alias("w_naive_median"),
                pl.col("w_h").std().alias("w_naive_std"),
            ])
            .join(
                cluster_stats.select([
                    "cluster_id",
                    "w_c",
                    "poly_ratio",
                    "version_ratio"
                ]),
                on="cluster_id"
            )
            .with_columns([
                (pl.col("w_naive_mean") / pl.col("w_c")).alias("inflation_ratio_mean"),
                (pl.col("w_naive_median") / pl.col("w_c")).alias("inflation_ratio_median"),

                # NEW (very useful in paper)
                (pl.col("w_naive_std") / pl.col("w_c")).alias("inflation_variance_ratio"),
            ])
        )


        # 7. SAVE RESULTS

        out_path = Path("results/third_experiment.parquet")
        comparison_df.sink_parquet(out_path)

        df = pd.read_parquet(out_path)

        print("[✓] Third experiment completed: structural vs naive weighting comparison")

        return df



    # EXECUTION

    from cloneage import run_louvain_clustering

    threshold_skeleton = 8

    vertex_to_cluster, modularity = run_louvain_clustering(
        EDGE_DIR / "part_*.parquet",
        threshold=threshold_skeleton
    )

    N_total = (
        pl.scan_csv(EXTENSIONS_PATH)
        .select("id")
        .unique()
        .collect()
        .height
    )

    stats = generate_cluster_stats(vertex_to_cluster, N_total)

    return stats


def fourth_experiment():
    """
    Experiment 4 — Extension Representation and Similarity.

    Validates:
        A) Robustness against extension size imbalance
        B) Baseline similarity comparisons
        C) Graph sparsity / discriminative power
        D) Family coherence
        E) Structural compression efficiency (H_X -> C_X)

    Outputs:
        - experiment_4_pairwise_scores.parquet
        - experiment_4_metrics.csv
    """

    print("[+] Running Experiment 4: Structural Validation of Extension Similarity")


    # CONFIG

    threshold_skeleton = 8
    tau = 0.9

    profiles_path = RESULTS_PATH / f"extension_profiles_skel_{threshold_skeleton}.parquet"
    weights_path = RESULTS_PATH / f"cluster_weights_skel_{threshold_skeleton}.parquet"
    lineages_path = RESULTS_PATH / f"extension_lineages_skel_{threshold_skeleton}_tau_{tau}.parquet"

    if not profiles_path.exists():
        raise FileNotFoundError(profiles_path)

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    if not lineages_path.exists():
        raise FileNotFoundError(lineages_path)

  
    # LOAD DATA
  

    profiles = pl.scan_parquet(profiles_path)
    weights = pl.scan_parquet(weights_path)
    lineages = pl.scan_parquet(lineages_path)

  
    # BUILD EXTENSION REPRESENTATIONS
  

    print("[+] Building extension representations...")

    df = (
        profiles
        .join(
            weights.select(["cluster_id", "w_c"]),
            on="cluster_id",
            how="left"
        )
    )

  
    # Extension totals
  

    ext_totals = (
        df.group_by("extension_id")
        .agg([
            pl.col("w_c").sum().alias("W_X"),
            pl.col("cluster_id").n_unique().alias("N_X")
        ])
    )

  
    # BUILD PAIRS
  

    print("[+] Building candidate extension pairs...")

    pairs = (
        df.select(["extension_id", "cluster_id", "w_c"])
        .join(
            df.select(["extension_id", "cluster_id", "w_c"]),
            on="cluster_id",
            suffix="_right"
        )
        .filter(
            pl.col("extension_id") < pl.col("extension_id_right")
        )
    )


    # INTERSECTIONS


    print("[+] Computing pairwise intersections...")

    intersections = (
        pairs.group_by(["extension_id", "extension_id_right"])
        .agg([
            pl.col("w_c").sum().alias("W_M"),
            pl.col("cluster_id").n_unique().alias("shared_clusters")
        ])
    )


    # FINAL SCORES


    print("[+] Computing similarity metrics...")

    results = (
        intersections

        # LEFT EXT
        .join(
            ext_totals,
            on="extension_id"
        )

        # RIGHT EXT
        .join(
            ext_totals,
            left_on="extension_id_right",
            right_on="extension_id",
            suffix="_B"
        )

        .with_columns([
            pl.col("W_X").alias("W_A"),
            pl.col("W_X_B").alias("W_B"),

            pl.col("N_X").alias("N_A"),
            pl.col("N_X_B").alias("N_B")
        ])


        # Dynamic Tversky


        .with_columns([
            (
                pl.col("W_A") /
                (pl.col("W_A") + pl.col("W_B"))
            ).alias("alpha"),

            (
                pl.col("W_B") /
                (pl.col("W_A") + pl.col("W_B"))
            ).alias("beta")
        ])

        .with_columns([


            (
                pl.col("W_M") / (
                    pl.col("W_M")
                    + (
                        pl.col("alpha")
                        * (pl.col("W_A") - pl.col("W_M"))
                    )
                    + (
                        pl.col("beta")
                        * (pl.col("W_B") - pl.col("W_M"))
                    )
                )
            ).alias("score_tversky_dynamic"),


            # Baseline 1 — Unweighted Jaccard


            (
                pl.col("shared_clusters") /
                (
                    pl.col("N_A")
                    + pl.col("N_B")
                    - pl.col("shared_clusters")
                )
            ).alias("score_jaccard"),


            # Baseline 2 — Weighted Overlap


            (
                pl.col("W_M") /
                pl.min_horizontal(["W_A", "W_B"])
            ).alias("score_overlap"),


            # Size imbalance


            (
                pl.max_horizontal(["W_A", "W_B"])
                /
                pl.min_horizontal(["W_A", "W_B"])
            ).alias("size_imbalance_ratio")
        ])
        .collect(engine="streaming")
    )


    # ROBUSTNESS ANALYSIS


    print("[+] Running robustness analysis...")

    corr_spearman, pvalue = spearmanr(
        results["size_imbalance_ratio"].to_numpy(),
        results["score_tversky_dynamic"].to_numpy()
    )


    # GRAPH SPARSITY


    print("[+] Computing graph sparsity...")

    N_total = (
        pl.scan_csv(EXTENSIONS_PATH)
        .select("id")
        .unique()
        .collect()
        .height
    )

    possible_edges = (N_total * (N_total - 1)) / 2

    actual_edges = results.height

    sparsity = 1 - (actual_edges / possible_edges)


    # FAMILY COHERENCE


    print("[+] Computing family coherence...")

    from cloneage import detect_extension_families

    family_df, _ = detect_extension_families(
        lineages.collect()
    )

    fam_A = family_df.rename({
        "extension_id": "extension_id",
        "family_id": "family_A"
    })

    fam_B = family_df.rename({
        "extension_id": "extension_id_right",
        "family_id": "family_B"
    })

    coherence_df = (
        results.lazy()

        .join(fam_A.lazy(), on="extension_id")
        .join(fam_B.lazy(), on="extension_id_right")

        .filter(
            pl.col("family_A") == pl.col("family_B")
        )

        .group_by("family_A")
        .agg([
            pl.col("score_tversky_dynamic")
            .mean()
            .alias("internal_coherence"),

            pl.len().alias("n_edges")
        ])

        .filter(pl.col("n_edges") > 1)

        .collect()
    )

    avg_family_coherence = (
        coherence_df["internal_coherence"].mean()
    )


    # STRUCTURAL COMPRESSION


    print("[+] Computing structural compression ratios...")

    raw_per_ext = (
        pl.scan_csv(EXTLIB_PATH)
        .group_by("ext_id")
        .agg(
            pl.col("lib_id")
            .n_unique()
            .alias("raw_hashes")
        )
    )

    clusters_per_ext = (
        pl.scan_parquet(profiles_path)
        .group_by("extension_id")
        .agg(
            pl.col("cluster_id")
            .n_unique()
            .alias("cluster_nodes")
        )
    )

    compression = (
        raw_per_ext
        .join(
            clusters_per_ext,
            left_on="ext_id",
            right_on="extension_id"
        )
        .with_columns([
            (
                1 - (
                    pl.col("cluster_nodes")
                    / pl.col("raw_hashes")
                )
            ).alias("compression_ratio")
        ])
        .collect()
    )

    compression_mean = compression["compression_ratio"].mean()
    compression_median = compression["compression_ratio"].median()
    compression_p90 = compression["compression_ratio"].quantile(0.90)


    # BASELINE CONTRAST


    print("[+] Computing baseline contrast metrics...")

    results = results.with_columns([

        (
            pl.col("score_tversky_dynamic")
            - pl.col("score_jaccard")
        ).alias("delta_vs_jaccard"),

        (
            pl.col("score_tversky_dynamic")
            - pl.col("score_overlap")
        ).alias("delta_vs_overlap")
    ])


    # SAVE PAIRWISE RESULTS


    out_pairs = RESULTS_PATH / "experiment_4_pairwise_scores.parquet"

    results.write_parquet(out_pairs)


    # FINAL METRICS TABLE


    metrics = pl.DataFrame({

        "metric": [

            # robustness
            "spearman_imbalance_vs_score",
            "spearman_pvalue",

            # graph
            "graph_sparsity",

            # families
            "avg_family_coherence",

            # compression
            "compression_mean",
            "compression_median",
            "compression_p90",

            # similarity baselines
            "mean_tversky_dynamic",
            "mean_jaccard",
            "mean_overlap",

            "delta_vs_jaccard",
            "delta_vs_overlap"
        ],

        "value": [

            # robustness
            corr_spearman,
            pvalue,

            # graph sparsity
            sparsity,

            # families
            avg_family_coherence,

            # compression
            compression_mean,
            compression_median,
            compression_p90,

            # scores
            results["score_tversky_dynamic"].mean(),
            results["score_jaccard"].mean(),
            results["score_overlap"].mean(),

            results["delta_vs_jaccard"].mean(),
            results["delta_vs_overlap"].mean()
        ]
    })

    metrics_path = RESULTS_PATH / "experiment_4_metrics.csv"

    metrics.write_csv(metrics_path)

    
    # REPORT
    

    print("\n==============================")
    print("EXPERIMENT 4 RESULTS")
    print("==============================")

    print(f"[*] Spearman(Imbalance vs Score): {corr_spearman:.4f}")
    print(f"[*] p-value: {pvalue:.6f}")

    print(f"[*] Graph Sparsity: {sparsity:.6%}")

    print(f"[*] Average Family Coherence: {avg_family_coherence:.4f}")

    print(f"[*] Compression Mean: {compression_mean:.2%}")
    print(f"[*] Compression Median: {compression_median:.2%}")
    print(f"[*] Compression P90: {compression_p90:.2%}")

    print(f"[*] Mean Tversky Dynamic: {results['score_tversky_dynamic'].mean():.4f}")
    print(f"[*] Mean Jaccard: {results['score_jaccard'].mean():.4f}")
    print(f"[*] Mean Overlap: {results['score_overlap'].mean():.4f}")

    print(f"[*] Δ vs Jaccard: {results['delta_vs_jaccard'].mean():.4f}")
    print(f"[*] Δ vs Overlap: {results['delta_vs_overlap'].mean():.4f}")

    print("\n[✓] Experiment 4 completed.")

    return metrics


def compute_family_stability(family_mappings_by_tau):
    import pandas as pd

    taus = sorted(family_mappings_by_tau.keys())
    rows = []

    for i in range(1, len(taus)):
        t0, t1 = taus[i-1], taus[i]

        prev = family_mappings_by_tau[t0]
        curr = family_mappings_by_tau[t1]

        # FORCE SAFE MERGE
        merged = prev.merge(
            curr,
            on="extension_id",
            suffixes=("_prev", "_curr")
        )

        if merged.empty:
            rows.append({
                "tau_prev": t0,
                "tau_curr": t1,
                "avg_best_match": 0.0,
                "min_stability": 0.0,
                "std_stability": 0.0
            })
            continue

        cross = (
            merged
            .groupby(["family_id_prev", "family_id_curr"])
            .size()
            .reset_index(name="n")
        )

        prev_sizes = merged["family_id_prev"].value_counts()
        curr_sizes = merged["family_id_curr"].value_counts()

        cross["prev_size"] = cross["family_id_prev"].map(prev_sizes)
        cross["curr_size"] = cross["family_id_curr"].map(curr_sizes)

        cross["jaccard"] = cross["n"] / (
            cross["prev_size"] +
            cross["curr_size"] -
            cross["n"]
        )

        best = cross.groupby("family_id_prev")["jaccard"].max()

        rows.append({
            "tau_prev": t0,
            "tau_curr": t1,
            "avg_best_match": float(best.mean()),
            "min_stability": float(best.min()),
            "std_stability": float(best.std())
        })

    return pd.DataFrame(rows)


### Fifth Experiment ###

def evaluate_extension_family_formation(
        ext_profiles_path: str,
        cluster_weights_path: str,
        tau_values=(0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
        min_wc: float = 0.0,
        output_dir: str = "results/fifth_experiment"
):
    """
    Fifth Experiment: Extension Family Formation Evaluation

    Evaluates the final stage of CLONEAGE:
        1. Builds extension similarity graph dynamically per threshold.
        2. Applies Jaccard/Tversky thresholding based on tau_ext.
        3. Detects connected components using igraph (C-backend).
        4. Measures global graph structural topologies (Density, GCR).
        5. Computes family-size distribution profiles.
        6. Tracks evolutionary lineage stability across partitions.
        7. Exports isolated Gephi-ready files.
    """
    from cloneage import compute_extension_similarity

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("FIFTH EXPERIMENT — EXTENSION FAMILY FORMATION")
    print("=" * 80)

    
    # LOAD INPUTS
    
    print("[#] Loading infrastructure profiles...")
    ext_profiles = pl.read_parquet(ext_profiles_path)
    cluster_weights = pl.read_parquet(cluster_weights_path)

    sensitivity_rows = []
    family_mappings_by_tau = {}

    # Load security labels
    print("[#] Loading ground truth security labels from datasets...")
    alive_df = (
        pl.read_csv("datasets/extensions.csv")
        .select([
            pl.col("id").alias("extension_id"),
            pl.col("alive"),
            pl.col("in_cws"),
            pl.col("downloadable"),
        ])
    )
    malware_df = (
        pl.read_csv("datasets/extensions.csv")
        .select([
            pl.col("id").alias("extension_id"),
            pl.col("obsoleteReason"),
            (
                    pl.col("obsoleteReason")
                    .is_not_null()
                    & (pl.col("obsoleteReason").str.strip_chars() != "")
            ).alias("is_malware")
        ])
    )


    # TAU SENSITIVITY LOOP

    for tau_ext in tau_values:

        tau_tag = f"{tau_ext:.2f}".replace(".", "_")
        minwc_tag = f"{min_wc:.2f}".replace(".", "_")

        experiment_dir = output_dir / f"tau_{tau_tag}_minwc_{minwc_tag}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 80)
        print(f"[#] Evaluating tau_ext = {tau_ext:.2f}")
        print("-" * 80)


        # COMPUTE EXTENSION SIMILARITIES

        print("[+] Computing extension similarities via Tversky...")
        similarities = compute_extension_similarity(
            ext_profiles.lazy(),
            cluster_weights.lazy(),
            threshold=tau_ext,
            min_wc=min_wc
        )

        if similarities.height == 0:
            print(f"[!] No similarity edges found for threshold tau = {tau_ext:.2f}")
            continue


        # NORMALIZE COLUMN NAMES

        similarities = similarities.select([
            pl.col("extension_id").cast(pl.Utf8).alias("source"),
            pl.col("extension_id_right").cast(pl.Utf8).alias("target"),
            pl.col("score").cast(pl.Float64).alias("weight")
        ])

        print(f"[✓] Similarity edges retained: {similarities.height:,}")


        # SAVE RAW EDGES

        similarities.write_parquet(experiment_dir / "extension_similarity_edges.parquet")


        # BUILD GRAPH (via optimized igraph TupleList)

        edge_pd = similarities.to_pandas()

        g = ig.Graph.TupleList(
            edge_pd.itertuples(index=False),
            directed=False,
            weights=True,
            vertex_name_attr="name"
        )

        v_count = g.vcount()
        e_count = g.ecount()
        print(f"[✓] Graph built successfully: {v_count:,} nodes / {e_count:,} edges")


        # CONNECTED COMPONENTS

        components = g.connected_components()
        family_sizes = np.array([len(c) for c in components])

        n_families = len(family_sizes)
        largest_family = int(family_sizes.max()) if n_families > 0 else 0
        mean_family_size = float(np.mean(family_sizes)) if n_families > 0 else 0.0
        median_family_size = float(np.median(family_sizes)) if n_families > 0 else 0.0

        giant_component_ratio = largest_family / v_count if v_count > 0 else 0.0
        singleton_ratio = np.sum(family_sizes == 1) / n_families if n_families > 0 else 0.0
        density = g.density()

        print(f"[✓] Families detected: {n_families:,}")
        print(f"[✓] Largest family campaign size: {largest_family:,}")
        print(f"[✓] Mean family size: {mean_family_size:.2f}")
        print(f"[✓] Graph density: {density:.8f}")


        # STORE GLOBAL METRICS

        sensitivity_rows.append({
            "tau_ext": tau_ext,
            "nodes": v_count,
            "edges": e_count,
            "families": n_families,
            "largest_family": largest_family,
            "giant_component_ratio": giant_component_ratio,
            "singleton_ratio": singleton_ratio,
            "mean_family_size": mean_family_size,
            "median_family_size": median_family_size,
            "density": density
        })


        # FAMILY SUMMARY EXTRACTION

        family_rows = [
            {"family_id": f"fam_{idx}", "family_size": len(comp)}
            for idx, comp in enumerate(components)
        ]

        families_df = pl.DataFrame(family_rows)
        families_df.write_parquet(experiment_dir / "families_summary.parquet")

        total_families = families_df.height
        total_extensions = families_df["family_size"].sum() if total_families > 0 else 0


        # FAMILY SIZE DISTRIBUTION PROFILE

        if total_families > 0:
            distribution = (
                families_df
                .with_columns(
                    pl.when(pl.col("family_size") == 1).then(pl.lit("1"))
                    .when(pl.col("family_size") == 2).then(pl.lit("2"))
                    .when(pl.col("family_size") == 3).then(pl.lit("3"))
                    .when(pl.col("family_size") == 4).then(pl.lit("4"))
                    .when(pl.col("family_size") == 5).then(pl.lit("5"))
                    .when(pl.col("family_size").is_between(6, 10)).then(pl.lit("6--10"))
                    .when(pl.col("family_size").is_between(11, 50)).then(pl.lit("11--50"))
                    .when(pl.col("family_size").is_between(51, 100)).then(pl.lit("51--100"))
                    .when(pl.col("family_size").is_between(101, 200)).then(pl.lit("101--200"))
                    .when(pl.col("family_size").is_between(201, 300)).then(pl.lit("201--300"))
                    .when(pl.col("family_size").is_between(301, 500)).then(pl.lit("301--500"))
                    .otherwise(pl.lit("500+"))
                    .alias("bin_size")
                )
                .group_by("bin_size")
                .agg([
                    pl.len().alias("families"),
                    pl.col("family_size").sum().alias("extensions")
                ])
                .with_columns([
                    (pl.col("families") / total_families * 100).round(2).alias("families_pct"),
                    (pl.col("extensions") / total_extensions * 100).round(2).alias("extensions_pct")
                ])
            )
            distribution.write_csv(experiment_dir / "family_size_distribution.csv")
            print("[✓] Saved family size structural distribution.")

        # EXTENSION TO FAMILY NODE MAPPING & SECURITY JOIN

        mapping_rows = []
        for idx, comp in enumerate(components):
            family_id = f"fam_{idx}"
            comp_len = len(comp)
            for node in comp:
                mapping_rows.append({
                    "extension_id": g.vs[node]["name"],
                    "family_id": family_id,
                    "family_size": comp_len
                })


        mapping_df = (
            pl.DataFrame(mapping_rows)
            .join(alive_df, on="extension_id", how="left")
            .join(malware_df, on="extension_id", how="left")
        )
        mapping_df.write_parquet(experiment_dir / "extension_family_mapping.parquet")
        print("[✓] Saved node-level family operational mappings with security metadata.")


        family_mappings_by_tau[tau_ext] = mapping_df.select([
            "extension_id",
            "family_id",
            "family_size"
        ]).to_pandas()


        # EXPORT FAMILY SECURITY RATIOS

        family_stats = (
            mapping_df.group_by("family_id")
            .agg([
                pl.len().alias("size"),
                pl.col("is_malware").mean().alias("malware_ratio"),
                pl.col("alive").mean().alias("alive_ratio")
            ])
        )
        family_stats.write_csv(experiment_dir / "family_security_stats.csv")
        print("[✓] Saved aggregated family operational security profiles.")


        # GEPHI COMPATIBLE EDGE LIST

        gephi_edges = similarities.rename({
            "source": "Source",
            "target": "Target",
            "weight": "Weight"
        })
        gephi_edges.write_csv(experiment_dir / "gephi_edges.csv")
        print("[✓] Saved Gephi-ready network structural file.")


    # SAVE GLOBAL REGIMES (SENSITIVITY ANALYSIS)

    sensitivity_df = pl.DataFrame(sensitivity_rows)
    minwc_file_tag = f"{min_wc:.2f}".replace(".", "_")

    sensitivity_df.write_csv(
        output_dir / f"tau_sensitivity_minwc_{minwc_file_tag}.csv"
    )
    print("\n[✓] Compiled global structural parameter-sweep metrics.")


    # TRACK CROSS-THRESHOLD STABILITY

    if len(family_mappings_by_tau) > 1:
        print("[+] Processing evolutionary cross-threshold lineage tracking...")
        stability_df = compute_family_stability(family_mappings_by_tau)
        stability_df.to_csv(output_dir / f"lineage_stability_minwc_{minwc_file_tag}.csv")
        print("[✓] Exported programmatic stability and monotonicity reports.")
        print(stability_df)

    print("\n" + "=" * 80)
    print("[✓] EXPERIMENT PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 80)

    return sensitivity_df



# UNCOMMENT ANY EXPERIMENTS TO RUN
if __name__ == "__main__":
    print()
    # first_experiment()
    # second_experiment()
    # clustering_evaluation()
    # third_experiment()
    # fourth_experiment()


# Fifth experiment:
    results = evaluate_extension_family_formation(
        ext_profiles_path="results/extension_profiles_skel_8.parquet",
        cluster_weights_path="results/cluster_weights_skel_8.parquet",
        tau_values=[0.60, 0.70, 0.80, 0.90, 0.95, 0.99],
        min_wc=0.0
    )
    print(results)

    print("\n[#] Final Parameter Sweep Execution Summary:")