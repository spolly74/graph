"""Draft a SKOS vocabulary for the Azure Machine Learning docs corpus.

One-time drafting aid. The output (vocab/vocabulary.ttl) is owned by Scott
and is expected to be edited by hand after it is generated.

Usage:
    python tools/draft_vocabulary.py <path to articles/machine-learning> <output .ttl>

Hand-curated concepts live in CURATED below. Designer components, built-in
roles, CLI v2 YAML schemas, and prompt flow tools are extracted from the
corpus so that their labels match the source text exactly.

Front matter values such as ms.subservice are left out on purpose: they are
document metadata for provenance, not entities to match in text.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

BASE = "https://example.org/ckg/vocab/"  # TODO(Scott): choose a real namespace

# scheme id -> (label, intended domain class)
SCHEMES = {
    "platform": ("Platforms and products", "Product"),
    "resource": ("Azure Machine Learning resources", "Resource"),
    "compute": ("Compute types", "ComputeTarget"),
    "endpoint": ("Endpoint types", "Endpoint"),
    "deployment": ("Deployment types", "Deployment"),
    "asset": ("Asset types", "Asset"),
    "data_type": ("Data types", "DataType"),
    "job": ("Job types", "Job"),
    "model_type": ("Model types and formats", "ModelType"),
    "automl_task": ("AutoML task types", "AutomlTask"),
    "feature_store": ("Managed feature store entities", "FeatureStoreEntity"),
    "azure_service": ("Associated Azure services", "AzureService"),
    "security": ("Security and networking concepts", "SecurityConcept"),
    "identity": ("Identity and access concepts", "IdentityConcept"),
    "role": ("Built-in roles", "Role"),
    "interface": ("Developer interfaces and tools", "Interface"),
    "capability": ("Product capabilities", "Capability"),
    "designer_component": ("Designer components", "DesignerComponent"),
    "yaml_schema": ("CLI v2 YAML schemas", "YamlSchema"),
    "promptflow_tool": ("Prompt flow tools", "PromptFlowTool"),
}

G = "azure-machine-learning-glossary.md"

# (scheme, id, prefLabel, [altLabels], source article or None, broader id or None)
CURATED: list[tuple[str, str, str, list[str], str | None, str | None]] = [
    # Platforms
    ("platform", "azure-machine-learning", "Azure Machine Learning", ["Azure ML", "AzureML", "AML", "Machine Learning"], "overview-what-is-azure-machine-learning.md", None),
    ("platform", "azure-machine-learning-studio", "Azure Machine Learning studio", ["Machine Learning studio"], "overview-what-is-azure-machine-learning.md", None),
    ("platform", "microsoft-foundry", "Microsoft Foundry", ["Foundry", "Azure AI Foundry", "Azure AI Studio"], "overview-what-is-azure-machine-learning.md", None),
    ("platform", "prompt-flow", "prompt flow", ["promptflow"], "prompt-flow/overview-what-is-prompt-flow.md", None),
    ("platform", "microsoft-agent-framework", "Microsoft Agent Framework", ["Agent Framework"], "prompt-flow/migrate-prompt-flow-to-agent-framework.md", None),
    ("platform", "mlflow", "MLflow", [], "concept-mlflow.md", None),
    ("platform", "microsoft-fabric", "Microsoft Fabric", ["Fabric"], "data-import-migration-guide.md", None),
    ("platform", "onnx", "ONNX", ["Open Neural Network Exchange", "ONNX Runtime"], "concept-onnx.md", None),
    ("platform", "apache-spark", "Apache Spark", ["Spark"], "apache-spark-azure-ml-concepts.md", None),
    ("platform", "model-catalog", "model catalog", [], "foundry-models-overview.md", None),
    ("platform", "hugging-face-hub", "Hugging Face Hub", ["Hugging Face"], None, None),
    # Resources
    ("resource", "workspace", "workspace", ["Azure Machine Learning workspace", "Machine Learning workspace"], G, None),
    ("resource", "hub-workspace", "hub workspace", [], "concept-hub-workspace.md", "workspace"),
    ("resource", "project-workspace", "project workspace", [], "concept-hub-workspace.md", "workspace"),
    ("resource", "registry", "registry", ["Azure Machine Learning registry", "machine learning registry"], "concept-machine-learning-registries-mlops.md", None),
    ("resource", "datastore", "datastore", ["data store"], G, None),
    ("resource", "connection", "connection", ["workspace connection"], "how-to-connection.md", None),
    ("compute", "compute", "compute", ["compute target", "compute resource"], G, None),
    ("endpoint", "endpoint", "endpoint", [], "concept-endpoints.md", None),
    ("deployment", "deployment", "deployment", [], "concept-endpoints.md", None),
    ("resource", "schedule", "schedule", [], "how-to-schedule-pipeline-job.md", None),
    ("resource", "model-monitor", "model monitor", ["model monitoring"], "concept-model-monitoring.md", None),
    # Compute
    ("compute", "compute-cluster", "compute cluster", ["Azure Machine Learning compute cluster", "AmlCompute", "training cluster"], G, "compute"),
    ("compute", "compute-instance", "compute instance", ["Azure Machine Learning compute instance", "ComputeInstance"], G, "compute"),
    ("compute", "serverless-compute", "serverless compute", ["Azure Machine Learning serverless compute"], "how-to-use-serverless-compute.md", "compute"),
    ("compute", "kubernetes-compute", "Kubernetes compute", ["Kubernetes compute target", "KubernetesCompute", "Azure Machine Learning Kubernetes", "Kubernetes cluster"], "how-to-attach-kubernetes-anywhere.md", "compute"),
    ("compute", "attached-compute", "attached compute", [], G, "compute"),
    ("compute", "remote-vm", "remote VM", ["attached virtual machine", "remote virtual machine"], "concept-compute-target.md", "attached-compute"),
    ("compute", "synapse-spark-pool", "Synapse Spark pool", ["Apache Spark pool", "attached Synapse Spark pool"], "how-to-manage-synapse-spark-pool.md", "attached-compute"),
    ("compute", "serverless-spark-compute", "serverless Spark compute", ["serverless Spark"], "apache-spark-azure-ml-concepts.md", "compute"),
    ("compute", "azure-databricks-compute", "Azure Databricks", ["Databricks"], "concept-compute-target.md", "attached-compute"),
    ("compute", "azure-hdinsight", "Azure HDInsight", ["HDInsight"], "concept-compute-target.md", "attached-compute"),
    ("compute", "azure-batch", "Azure Batch", [], "concept-compute-target.md", "attached-compute"),
    ("compute", "azure-data-lake-analytics", "Azure Data Lake Analytics", [], "concept-compute-target.md", "attached-compute"),
    ("compute", "compute-session", "compute session", ["prompt flow compute session"], "prompt-flow/concept-session.md", None),
    ("compute", "instance-type", "instance type", [], "how-to-manage-kubernetes-instance-types.md", None),
    # Endpoints
    ("endpoint", "online-endpoint", "online endpoint", ["real-time endpoint"], "concept-endpoints-online.md", "endpoint"),
    ("endpoint", "managed-online-endpoint", "managed online endpoint", [], "concept-endpoints-online.md", "online-endpoint"),
    ("endpoint", "kubernetes-online-endpoint", "Kubernetes online endpoint", [], "concept-endpoints-online.md", "online-endpoint"),
    ("endpoint", "batch-endpoint", "batch endpoint", [], "concept-endpoints-batch.md", "endpoint"),
    ("endpoint", "serverless-api-endpoint", "serverless API endpoint", ["serverless API", "standard deployment", "serverless API deployment"], "how-to-deploy-models-serverless.md", "endpoint"),
    ("deployment", "online-deployment", "online deployment", [], "concept-endpoints-online.md", "deployment"),
    ("deployment", "batch-deployment", "batch deployment", [], "concept-endpoints-batch.md", "deployment"),
    ("deployment", "pipeline-component-deployment", "pipeline component deployment", [], "concept-endpoints-batch.md", "batch-deployment"),
    ("asset", "deployment-template", "deployment template", [], "concept-deployment-template.md", None),
    ("resource", "inference-router", "inference router", ["azureml-fe"], "how-to-kubernetes-inference-routing-azureml-fe.md", None),
    ("asset", "scoring-script", "scoring script", ["entry script", "score.py"], "how-to-deploy-online-endpoints.md", None),
    ("capability", "safe-rollout", "safe rollout", ["blue-green deployment", "mirrored traffic", "traffic mirroring"], "how-to-safely-rollout-online-endpoints.md", None),
    # Assets
    ("asset", "data-asset", "data asset", ["dataset"], "concept-data.md", None),
    ("asset", "model", "model", ["registered model", "model asset"], G, None),
    ("asset", "environment", "environment", [], G, None),
    ("asset", "curated-environment", "curated environment", [], "resource-curated-environments.md", "environment"),
    ("asset", "custom-environment", "custom environment", [], G, "environment"),
    ("asset", "component", "component", [], G, None),
    ("asset", "pipeline-component", "pipeline component", [], "how-to-use-pipeline-component.md", "component"),
    ("asset", "feature-set", "feature set", ["featureset"], "concept-what-is-managed-feature-store.md", None),
    ("asset", "index", "index", ["vector index"], "how-to-create-vector-index.md", None),
    ("asset", "flow", "flow", ["prompt flow flow"], "prompt-flow/concept-flows.md", None),
    # Data types
    ("data_type", "uri-file", "uri_file", [], G, None),
    ("data_type", "uri-folder", "uri_folder", [], G, None),
    ("data_type", "mltable", "mltable", [], G, None),
    # Jobs
    ("job", "job", "job", [], "how-to-train-model.md", None),
    ("job", "command-job", "command job", [], "reference-yaml-job-command.md", "job"),
    ("job", "sweep-job", "sweep job", ["hyperparameter sweep", "hyperparameter tuning"], "reference-yaml-job-sweep.md", "job"),
    ("job", "pipeline-job", "pipeline job", ["pipeline", "machine learning pipeline", "ML pipeline"], "concept-ml-pipelines.md", "job"),
    ("job", "parallel-job", "parallel job", [], "reference-yaml-job-parallel.md", "job"),
    ("job", "spark-job", "Spark job", [], "reference-yaml-job-spark.md", "job"),
    ("job", "automl-job", "AutoML job", ["automated ML job"], "concept-automated-ml.md", "job"),
    ("job", "data-import", "data import", [], "how-to-import-data-assets.md", None),
    # Model types
    ("model_type", "custom-model", "custom_model", [], G, None),
    ("model_type", "mlflow-model", "mlflow_model", ["MLflow model"], G, None),
    ("model_type", "triton-model", "triton_model", ["Triton model"], G, None),
    # AutoML tasks (labels follow component-reference-v2)
    ("automl_task", "classification", "Classification", [], "concept-automated-ml.md", None),
    ("automl_task", "regression", "Regression", [], "concept-automated-ml.md", None),
    ("automl_task", "forecasting", "Forecasting", ["time-series forecasting", "time series forecasting"], "concept-automated-ml.md", None),
    ("automl_task", "image-classification", "Image Classification", [], "how-to-auto-train-image-models.md", None),
    ("automl_task", "image-classification-multilabel", "Image Classification Multilabel", [], "how-to-auto-train-image-models.md", None),
    ("automl_task", "image-object-detection", "Image Object Detection", ["object detection"], "how-to-auto-train-image-models.md", None),
    ("automl_task", "image-instance-segmentation", "Image Instance Segmentation", ["instance segmentation"], "how-to-auto-train-image-models.md", None),
    ("automl_task", "text-classification", "Text Classification", [], "how-to-auto-train-nlp-models.md", None),
    ("automl_task", "text-classification-multilabel", "Text Classification Multilabel", [], "how-to-auto-train-nlp-models.md", None),
    ("automl_task", "text-ner", "Text NER", ["Named Entity Recognition", "NER"], "how-to-auto-train-nlp-models.md", None),
    # Feature store
    ("feature_store", "feature-store", "feature store", ["managed feature store"], "concept-what-is-managed-feature-store.md", None),
    ("feature_store", "feature-store-entity", "feature store entity", [], "concept-top-level-entities-in-managed-feature-store.md", None),
    ("feature_store", "materialization", "materialization", ["feature set materialization", "offline materialization", "online materialization"], "feature-set-materialization-concepts.md", None),
    ("feature_store", "feature-retrieval-specification", "feature retrieval specification", ["feature retrieval spec"], "feature-retrieval-concepts.md", None),
    ("feature_store", "point-in-time-join", "point-in-time join", [], "offline-retrieval-point-in-time-join-concepts.md", None),
    # Associated Azure services
    ("azure_service", "azure-storage-account", "Azure Storage account", ["storage account", "Azure Storage"], "concept-workspace.md", None),
    ("azure_service", "azure-blob-storage", "Azure Blob Storage", ["Blob Storage", "Azure Blob"], G, "azure-storage-account"),
    ("azure_service", "azure-files", "Azure Files", ["Azure file share"], G, "azure-storage-account"),
    ("azure_service", "azure-data-lake-storage-gen2", "Azure Data Lake Storage Gen2", ["ADLS Gen2", "Data Lake Storage Gen2"], G, None),
    ("azure_service", "azure-data-lake-storage-gen1", "Azure Data Lake Storage Gen1", ["ADLS Gen1", "Data Lake Storage Gen1"], G, None),
    ("azure_service", "azure-key-vault", "Azure Key Vault", ["Key Vault"], "concept-workspace.md", None),
    ("azure_service", "azure-container-registry", "Azure Container Registry", ["ACR"], "concept-workspace.md", None),
    ("azure_service", "application-insights", "Application Insights", ["Azure Application Insights"], "concept-workspace.md", None),
    ("azure_service", "azure-monitor", "Azure Monitor", [], "monitor-azure-machine-learning.md", None),
    ("azure_service", "log-analytics-workspace", "Log Analytics workspace", ["Log Analytics"], "monitor-azure-machine-learning.md", None),
    ("azure_service", "azure-kubernetes-service", "Azure Kubernetes Service", ["AKS"], "how-to-attach-kubernetes-anywhere.md", None),
    ("azure_service", "azure-arc-enabled-kubernetes", "Azure Arc-enabled Kubernetes", ["Arc Kubernetes", "Arc-enabled Kubernetes"], "how-to-attach-kubernetes-anywhere.md", None),
    ("azure_service", "azure-synapse-analytics", "Azure Synapse Analytics", ["Azure Synapse", "Synapse"], "how-to-manage-synapse-spark-pool.md", None),
    ("azure_service", "azure-openai", "Azure OpenAI", ["Azure OpenAI Service", "Azure OpenAI in Foundry Models"], None, None),
    ("azure_service", "azure-ai-search", "Azure AI Search", ["AI Search", "Azure Cognitive Search"], None, None),
    ("azure_service", "azure-ai-content-safety", "Azure AI Content Safety", ["Content Safety"], None, None),
    ("azure_service", "azure-policy", "Azure Policy", [], "how-to-integrate-azure-policy.md", None),
    ("azure_service", "azure-devops", "Azure DevOps", ["Azure Pipelines"], "how-to-devops-machine-learning.md", None),
    ("azure_service", "github-actions", "GitHub Actions", [], "how-to-github-actions-machine-learning.md", None),
    ("azure_service", "azure-event-grid", "Azure Event Grid", ["Event Grid"], "how-to-use-event-grid.md", None),
    ("azure_service", "azure-firewall", "Azure Firewall", [], "how-to-access-azureml-behind-firewall.md", None),
    ("azure_service", "azure-private-dns", "Azure Private DNS", ["private DNS zone"], "how-to-custom-dns.md", None),
    # Security and networking
    ("security", "managed-virtual-network", "managed virtual network", ["managed VNet", "workspace managed virtual network"], "how-to-managed-network.md", None),
    ("security", "allow-internet-outbound", "allow internet outbound", [], "how-to-managed-network.md", "managed-virtual-network"),
    ("security", "allow-only-approved-outbound", "allow only approved outbound", [], "how-to-managed-network.md", "managed-virtual-network"),
    ("security", "outbound-rule", "outbound rule", [], "how-to-managed-network.md", None),
    ("security", "azure-virtual-network", "Azure Virtual Network", ["virtual network", "VNet"], "how-to-network-security-overview.md", None),
    ("security", "private-endpoint", "private endpoint", ["Azure Private Link", "Private Link"], "how-to-configure-private-link.md", None),
    ("security", "service-tag", "service tag", [], "how-to-access-azureml-behind-firewall.md", None),
    ("security", "service-endpoint", "service endpoint", [], "how-to-network-security-overview.md", None),
    ("security", "network-security-group", "network security group", ["NSG"], "how-to-network-security-overview.md", None),
    ("security", "no-public-ip", "no public IP", [], "how-to-secure-training-vnet.md", None),
    ("security", "customer-managed-key", "customer-managed key", ["CMK", "customer-managed keys"], "concept-customer-managed-keys.md", None),
    ("security", "microsoft-managed-key", "Microsoft-managed key", ["Microsoft-managed keys"], "concept-data-encryption.md", None),
    ("security", "encryption-at-rest", "encryption at rest", [], "concept-data-encryption.md", None),
    ("security", "encryption-in-transit", "encryption in transit", [], "concept-data-encryption.md", None),
    ("security", "tls", "TLS", ["TLS/SSL", "SSL"], "how-to-secure-kubernetes-online-endpoint.md", None),
    ("security", "soft-delete", "soft delete", [], "concept-soft-delete.md", None),
    ("security", "disaster-recovery", "disaster recovery", ["failover", "business continuity"], "how-to-high-availability-machine-learning.md", None),
    ("security", "vulnerability-management", "vulnerability management", [], "concept-vulnerability-management.md", None),
    # Identity
    ("identity", "microsoft-entra-id", "Microsoft Entra ID", ["Azure Active Directory", "Azure AD", "Microsoft Entra"], "how-to-setup-authentication.md", None),
    ("identity", "managed-identity", "managed identity", ["MSI"], "how-to-identity-based-service-authentication.md", None),
    ("identity", "system-assigned-managed-identity", "system-assigned managed identity", ["system-assigned identity"], "how-to-identity-based-service-authentication.md", "managed-identity"),
    ("identity", "user-assigned-managed-identity", "user-assigned managed identity", ["user-assigned identity", "UAI"], "how-to-identity-based-service-authentication.md", "managed-identity"),
    ("identity", "service-principal", "service principal", [], "how-to-setup-authentication.md", None),
    ("identity", "azure-rbac", "Azure role-based access control", ["Azure RBAC", "RBAC", "role-based access control"], "how-to-assign-roles.md", None),
    ("identity", "custom-role", "custom role", [], "how-to-assign-roles.md", None),
    ("identity", "identity-based-access", "identity-based data access", ["identity-based authentication", "credential-less"], "how-to-administrate-data-authentication.md", None),
    ("identity", "credential-based-access", "credential-based data access", ["credential-based authentication"], "how-to-administrate-data-authentication.md", None),
    ("identity", "shared-key", "shared key", ["account key", "storage access key"], "how-to-disable-local-auth-storage.md", None),
    ("identity", "sas-token", "SAS token", ["shared access signature"], "how-to-datastore.md", None),
    ("identity", "key-auth", "key-based authentication", ["endpoint key"], "concept-endpoints-online-auth.md", None),
    ("identity", "aml-token", "Azure Machine Learning token", ["aml_token", "AML token"], "concept-endpoints-online-auth.md", None),
    ("identity", "entra-token", "Microsoft Entra token", ["aad_token", "Entra token"], "concept-endpoints-online-auth.md", None),
    # Interfaces
    ("interface", "cli-v2", "Azure Machine Learning CLI v2", ["CLI v2", "CLI (v2)", "ml extension", "az ml"], "concept-v2.md", None),
    ("interface", "python-sdk-v2", "Azure Machine Learning Python SDK v2", ["Python SDK v2", "SDK v2", "azure-ai-ml"], "concept-v2.md", None),
    ("interface", "rest-api", "REST API", ["Azure Machine Learning REST API"], "how-to-manage-rest.md", None),
    ("interface", "azure-portal", "Azure portal", [], None, None),
    ("interface", "azure-powershell", "Azure PowerShell", [], "how-to-manage-workspace-powershell.md", None),
    ("interface", "arm-template", "Azure Resource Manager template", ["ARM template", "Resource Manager template"], "how-to-create-workspace-template.md", None),
    ("interface", "bicep", "Bicep", ["Bicep template"], "tutorial-create-secure-workspace-vnet.md", None),
    ("interface", "terraform", "Terraform", [], "how-to-manage-workspace-terraform.md", None),
    ("interface", "vs-code-extension", "Azure Machine Learning extension for Visual Studio Code", ["VS Code extension", "Visual Studio Code extension"], "how-to-setup-vs-code.md", None),
    ("interface", "jupyter-notebook", "Jupyter notebook", ["Jupyter", "notebook"], "how-to-run-jupyter-notebooks.md", None),
    ("interface", "designer", "designer", ["Azure Machine Learning designer"], "concept-designer.md", None),
    ("interface", "azureml-extension-k8s", "Azure Machine Learning extension", ["AzureML extension", "Kubernetes extension"], "how-to-deploy-kubernetes-extension.md", None),
    # Capabilities
    ("capability", "automated-ml", "automated machine learning", ["automated ML", "AutoML"], "concept-automated-ml.md", None),
    ("capability", "distributed-training", "distributed training", [], "concept-distributed-training.md", None),
    ("capability", "data-labeling", "data labeling", ["labeling project", "data labeling project"], "how-to-create-image-labeling-projects.md", None),
    ("capability", "responsible-ai-dashboard", "Responsible AI dashboard", ["RAI dashboard"], "concept-responsible-ai-dashboard.md", None),
    ("capability", "responsible-ai-scorecard", "Responsible AI scorecard", ["RAI scorecard"], "concept-responsible-ai-scorecard.md", None),
    ("capability", "model-interpretability", "model interpretability", ["explainability"], "how-to-machine-learning-interpretability.md", None),
    ("capability", "fairness-assessment", "fairness assessment", ["fairness"], "concept-fairness-ml.md", None),
    ("capability", "error-analysis", "error analysis", [], "concept-error-analysis.md", None),
    ("capability", "counterfactual-analysis", "counterfactual analysis", ["what-if analysis"], "concept-counterfactual-analysis.md", None),
    ("capability", "causal-inference", "causal inference", ["causal analysis"], "concept-causal-inference.md", None),
    ("capability", "data-collection", "model data collection", ["data collector", "production data collection"], "concept-data-collection.md", None),
    ("capability", "retrieval-augmented-generation", "Retrieval Augmented Generation", ["RAG"], "concept-retrieval-augmented-generation.md", None),
    ("capability", "mlops", "MLOps", ["machine learning operations"], "concept-model-management-and-deployment.md", None),
    ("capability", "llmops", "LLMOps", ["GenAIOps"], "prompt-flow/concept-llmops-maturity.md", None),
    ("capability", "experiment-tracking", "experiment tracking", [], "how-to-use-mlflow-configure-tracking.md", None),
    ("capability", "git-integration", "Git integration", [], "concept-train-model-git-integration.md", None),
    ("capability", "quota", "quota", ["resource quota"], "how-to-manage-quotas.md", None),
    ("capability", "service-limit", "service limit", ["service limits"], "resource-limits-capacity.md", None),
    ("capability", "idle-shutdown", "idle shutdown", [], "how-to-create-compute-instance.md", None),
    ("capability", "autoscale", "autoscale", ["autoscaling"], "how-to-autoscale-endpoints.md", None),
]



def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def toc_leaf_names(toc: Path) -> list[tuple[str, str]]:
    """Return (name, href) for TOC entries that link to a component article."""
    out, name = [], None
    for line in toc.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*-?\s*name:\s*(.+)", line)
        if m:
            name = m.group(1).strip().strip("'\"")
            continue
        h = re.match(r"\s*href:\s*(\S+)", line)
        if h and name:
            out.append((name, h.group(1)))
            name = None
    return out


def extract(root: Path) -> list[tuple[str, str, str, list[str], str | None, str | None]]:
    rows = []
    skip = re.compile(r"^(component-reference|designer-error-codes|graph-search-syntax|\.\./)")
    for folder in ("component-reference", "component-reference-v2"):
        for name, href in toc_leaf_names(root / folder / "toc.yml"):
            if skip.match(href) or folder == "component-reference-v2":
                continue  # v2 entries are AutoML tasks, curated above
            rows.append(("designer_component", slug(name), name, [], f"{folder}/{href}", None))

    roles = sorted(set(re.findall(r"^\| \*\*([^*]+)\*\*", (root / "how-to-assign-roles.md").read_text(encoding="utf-8"), re.M)))
    for r in roles:
        rows.append(("role", "role-" + slug(r), r, [], "how-to-assign-roles.md", None))

    for f in sorted(root.glob("reference-yaml-*.md")):
        if f.stem in ("reference-yaml-overview", "reference-yaml-core-syntax"):
            continue
        m = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", f.read_text(encoding="utf-8"), re.M)
        if m:
            rows.append(("yaml_schema", "yaml-" + f.stem.removeprefix("reference-yaml-"), m.group(1), [], f.name, None))

    for f in sorted((root / "prompt-flow" / "tools-reference").glob("*-tool.md")):
        m = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", f.read_text(encoding="utf-8"), re.M)
        label = m.group(1) if m else f.stem
        rows.append(("promptflow_tool", "pf-" + f.stem, label, [], f"prompt-flow/tools-reference/{f.name}", None))

    return rows


def lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"@en'


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = CURATED + extract(root)

    ids: dict[str, str] = {}
    for scheme, cid, *_ in rows:
        if cid in ids:
            raise SystemExit(f"duplicate concept id {cid} in {scheme} and {ids[cid]}")
        ids[cid] = scheme
    for scheme, cid, _, _, _, broader in rows:
        if broader and broader not in ids:
            raise SystemExit(f"{cid}: unknown broader {broader}")

    lines = [
        "# Draft vocabulary for the Azure Machine Learning docs corpus.",
        f"# Generated {date.today().isoformat()} by tools/draft_vocabulary.py. Owned by Scott; edit freely.",
        "# ckg:domainClass on each scheme names the domain.yaml class its concepts map to (proposed, not decided).",
        "",
        f"@prefix v: <{BASE}> .",
        f"@prefix ckg: <{BASE}terms#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix dct: <http://purl.org/dc/terms/> .",
        "",
    ]
    for sid, (label, cls) in SCHEMES.items():
        lines += [f"v:scheme-{sid} a skos:ConceptScheme ;", f"    skos:prefLabel {lit(label)} ;", f'    ckg:domainClass "{cls}" .', ""]

    for scheme, cid, pref, alts, source, broader in rows:
        p = [f"v:{cid} a skos:Concept", f"    skos:inScheme v:scheme-{scheme}", f"    skos:prefLabel {lit(pref)}"]
        for a in dict.fromkeys(a for a in alts if a != pref):
            p.append(f"    skos:altLabel {lit(a)}")
        if broader:
            p.append(f"    skos:broader v:{broader}")
        else:
            p.append(f"    skos:topConceptOf v:scheme-{scheme}")
        if source:
            p.append(f'    dct:source "articles/machine-learning/{source}"')
        lines.append(" ;\n".join(p) + " .\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    counts: dict[str, int] = {}
    for scheme, *_ in rows:
        counts[scheme] = counts.get(scheme, 0) + 1
    print(f"{len(rows)} concepts -> {out}")
    for s, n in counts.items():
        print(f"  {s:20} {n}")


if __name__ == "__main__":
    main()
