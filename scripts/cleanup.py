

import argparse
from kfp.client import Client

KFP_HOST = "http://localhost:8080"


def list_runs(client, page_size=100):
    runs = client.list_runs(page_size=page_size)
    for r in runs.runs:
        print(f"{r.run_id}  {r.display_name}  {r.created_at}")
    return runs.runs


def delete_all_in_experiment(client, experiment_name):
    experiment = client.get_experiment(experiment_name=experiment_name)
    runs = client.list_runs(experiment_id=experiment.experiment_id, page_size=100)
    for r in runs.runs:
        print(f"Elimino run: {r.run_id} ({r.display_name})")
        client.delete_run(r.run_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Elenca tutte le run")
    parser.add_argument(
        "--delete-all-in-experiment",
        type=str,
        default=None,
        help="Elimina tutte le run di un dato esperimento",
    )
    args = parser.parse_args()

    client = Client(host=KFP_HOST)

    if args.list:
        list_runs(client)
    elif args.delete_all_in_experiment:
        delete_all_in_experiment(client, args.delete_all_in_experiment)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
