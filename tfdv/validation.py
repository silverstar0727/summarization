import tensorflow_data_validation as tfdv

def validate(train_path, eval_path):
    train_stats = tfdv.generate_statistics_from_csv(data_location=train_path)
    eval_stats = tfdv.generate_statistics_from_csv(data_location=eval_path)

    schema = tfdv.infer_schema(statistics = train_stats)
    anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

    return anomalies

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path")
    parser.add_argument("--eval_path")
    args = parser.parse_args()


    anomalies = validate(args.train_path, args.eval_path)

    if anomalies != None:
        raise TypeError()