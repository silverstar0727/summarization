import tensorflow_data_validation as tfdv

train_stats = tfdv.generate_statistics_from_csv(data_location='/content/sample_data/california_housing_train.csv')
eval_stats = tfdv.generate_statistics_from_csv(data_location='/content/sample_data/california_housing_test.csv')

schema = tfdv.infer_schema(statistics = train_stats)
anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

if anomalies != None:
    raise TypeError()