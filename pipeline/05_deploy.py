import os
import yaml

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from jinja2 import Template


os.environ["EXPERIMENT_ID"] = "1"
os.environ["MODEL_NAME"] = "seoulbike-rental-prediction-rf"
os.environ["MODEL_COORDINATES"] = "013596862746.dkr.ecr.ap-northeast-2.amazonaws.com/mlops/basic-model:latest"
os.environ["INGRESS_HOST"] = "model.cjhyun.people.aws.dev"

run_id = os.environ["EXPERIMENT_ID"]
model_name = os.environ["MODEL_NAME"]
model_container_location = os.environ["MODEL_COORDINATES"]
ingress_host = os.environ["INGRESS_HOST"]

def init_kube_config():
    config.load_incluster_config()

def apply_sheldon_from_file(namespace : str):
    template_data = { "experiment_id": run_id, 
                     "model_name": model_name, 
                     "model_coordinates": model_container_location, 
                     "ingress_host": ingress_host, 
                     "namespace" : namespace }
    seldon_template = Template(open("SeldonDeployment/SeldonDeploy.yaml").read())
    rendered_template = seldon_template.render(template_data)
    template_to_yaml = yaml.safe_load(rendered_template)

    api_instance = client.CustomObjectsApi()

    group = "machinelearning.seldon.io"
    version = "v1"
    plural = "seldondeployments"
    name = f"model-{run_id}"

    try:
        api_instance.create_namespaced_custom_object(group=group, version=version, plural=plural, namespace=namespace, body=template_to_yaml)                                                    
    except ApiException as e:
        if e.status == 409:
            api_instance.patch_namespaced_custom_object(group=group, version=version, plural=plural, 
                                                name=name, namespace=namespace, body=template_to_yaml)
        else:
            print(f"Error applying SeldonDeployment: {e}")
            raise

def apply_ingress(namespace : str):

    api_instance = client.NetworkingV1Api()

    ingress = client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(
            name="sheldon-ingress",
            namespace=namespace,
            annotations={
                "alb.ingress.kubernetes.io/scheme": "internet-facing",
                "alb.ingress.kubernetes.io/target-type": "ip",
                "alb.ingress.kubernetes.io/listen-ports": '[{"HTTP": 80}, {"HTTPS":443}]',
                "alb.ingress.kubernetes.io/ssl-redirect": '443',
                "alb.ingress.kubernetes.io/healthcheck-port": '8000',
                "alb.ingress.kubernetes.io/healthcheck-path": "/ready",
                "alb.ingress.kubernetes.io/success-codes": '200'
            },
        ),
        spec=client.V1IngressSpec(
        ingress_class_name="mlops-ingress-class",
        rules=[
            client.V1IngressRule(
                host=ingress_host,
                http=client.V1HTTPIngressRuleValue(
                    paths=[
                        client.V1HTTPIngressPath(
                            path="/",
                            path_type="Prefix",
                            backend=client.V1IngressBackend(
                                service=client.V1IngressServiceBackend(
                                    name=f"model-{run_id}-{model_name}",
                                    port=client.V1ServiceBackendPort(
                                        number=8000
                                    )
                                )
                            )
                        )
                    ]
                )
            )
        ]
        )
    )
  
    try:
        # 존재한다면 업데이트
        api_instance.create_namespaced_ingress(
            namespace=namespace,
            body=ingress
        )
        print(f"Ingress updated in namespace {namespace}")
    except ApiException as e:
        if e.status == 409:
            # 존재하지 않으면 생성
            api_instance.patch_namespaced_ingress(
                name="sheldon-ingress",
                namespace=namespace,
                body=ingress
            )
            print(f"Ingress created in namespace {namespace}")
        else:
            print(f"Exception when applying ingress: {e}")

# 사용 예시
if __name__ == "__main__":
    namespace = "default"
    init_kube_config()
    apply_sheldon_from_file(namespace)
    apply_ingress(namespace)