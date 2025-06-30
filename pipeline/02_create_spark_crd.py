import os
import yaml

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from jinja2 import Template

def init_kube_config():
    config.load_incluster_config()

def create_spark_from_file():
    with open("SparkApplications/spark-preprocessing.yaml") as f:
        spark_template = Template(f.read())
    
    spark_yaml = yaml.safe_load(spark_template.render())
    
    print("Rendered YAML content:")
    print(yaml.dump(spark_yaml))

    api_instance = client.CustomObjectsApi()

    group = "sparkoperator.k8s.io"
    version = "v1beta2"
    plural = "sparkapplications"
    name = "spark-preprocessing"
    namespace = "spark"
    try:
        api_instance.create_namespaced_custom_object(
            group=group, 
            version=version, 
            plural=plural, 
            namespace=namespace, 
            body=spark_yaml
        )
        print(f"SparkApplication '{name}' created successfully in namespace '{namespace}'")
    except ApiException as e:
        print(f"Exception when calling CustomObjectsApi->create_namespaced_custom_object: {e}")
        print(f"Response body: {e.body}")

if __name__ == "__main__":
    init_kube_config()
    create_spark_from_file()