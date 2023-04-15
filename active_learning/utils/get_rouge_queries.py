import argparse
from active_learning.al.selector import ActiveSelector
from active_learning.utils.general import json_dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rouges_dir", type=str, help="Path to directory with rouge scores"
    )
    parser.add_argument("--num_texts", type=int, help="Number of texts")
    parser.add_argument("--num_queries", type=int, help="Number of texts to query")
    parser.add_argument(
        "--ngramms_range", type=tuple, default=(1, 2), help="Strategy to use to query"
    )
    parser.add_argument(
        "--strategy", type=str, default="mmr_rec_prec", help="Strategy to use to query"
    )
    parser.add_argument(
        "--lamb", type=float, default=0.5, help="Lambda parameter value for MMR"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="query_ids.json",
        help="Path to save the queries",
    )

    args = parser.parse_args()
    selector = ActiveSelector(
        scores_dir=args.rouges_dir,
        idxs=args.num_texts,
        ngramms_range=args.ngramms_range,
        strategy=args.strategy,
        lamb=args.lamb,
    )
    query_ids = selector.query(args.num_queries)
    json_dump(query_ids, args.save_path)
