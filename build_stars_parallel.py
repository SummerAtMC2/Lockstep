from pyspark import SparkConf, SparkContext
import argparse
import time
import os

STAR_SIZE_LIMIT = 1


def configure_argparse():
    parser = argparse.ArgumentParser(description='Build lockstep stars')
    parser.add_argument('-i', '--input', default='s3a://netflow/*/')
    parser.add_argument('-o', '--output', default='hdfs://TrantorNS1/user/mpetriso/')
    parser.add_argument('-b', '--big_dt', default=100)
    parser.add_argument('-s', '--small_dt', default=10)
    return parser.parse_args()


# Configure Spark environment
def configure_spark():
    if 'SPARK_HOME' not in os.environ:
        os.environ['SPARK_HOME'] = '/opt/cloudera/parcels/CDH/lib/spark'

    conf = SparkConf().setAppName('lockstep_stars')
    conf.set("spark.executor.heartbeatInterval", "3600s")
    conf.set("spark.executor.memory", "1g")
    sc = SparkContext(conf=conf)

    sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", '*censored*')
    sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", '*censored*')
    sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", '*censored*')

    return sc


# Build stars
def build_stars(sc, input_s3_path, output_hdfs_path, big_dt, small_dt):
    # Dataset structure (example on first element):
    # netflow.first().split(";")[0] --> timestamp
    # netflow.first().split(";")[3] --> source IP
    # netflow.first().split(";")[5] --> destination IP
    netflow = sc.textFile(input_s3_path)
    netflow.cache()
    start_ts = time.time()

    # Timestamp when the netflow starts.
    first_ts_str = (netflow.first().split(";")[0])
    first_ts = epoch_ts(first_ts_str)

    # Timestamp when the netflow ends.
    last_ts_str = netflow.take(netflow.count()).split(";")[0]
    last_ts = epoch_ts(last_ts_str)

    while first_ts <= last_ts:
        # Get an RDD with the netflow in the interval [first_ts, first_ts + big_dt]
        interval = netflow.filter(lambda x: (epoch_ts(x.split(";")[0]) >= first_ts) and
                                            (epoch_ts(x.split(";")[0]) <= (first_ts + big_dt)))

        # Stars are created by using the groupByKey operation and have this format:
        # rhn = [lhn1, lhn2, lhn3, ...]
        # Currently right hand nodes will be source IPs, and left hand nodes destination IPs.
        # By interchanging the key and value in the lambda function you can obtain results for
        # rhn == destination IP and lhns == [source IPs].
        stars = interval.map(lambda line: (line.split(";")[3], line.split(";")[5])) \
            .groupByKey().map(lambda x: (x[0], list(x[1])))

        # Filter out small stars.
        stars.filter(lambda x: len(x[1]) > STAR_SIZE_LIMIT).collect()

        # Save stars on HDFS.
        stars.saveAsTextFile("{}_stars_interval_{}_{}".format(output_hdfs_path, first_ts, last_ts))

        # Move the interval down with the value of small delta t.
        first_ts += small_dt

    print("Elapsed time = ", time.time() - start_ts)


def epoch_ts(ts_string):
    pattern = '%Y-%m-%d %H:%M:%S.%f'
    return int(time.mktime(time.strptime(ts_string, pattern)))


if __name__ == '__main__':
    args = configure_argparse()
    spark_context = configure_spark()
    build_stars(spark_context, args.input, args.output, args.big_dt, args.small_dt)
