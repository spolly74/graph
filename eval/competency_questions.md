# Competency questions (draft for review)

Status: Draft, proposed by Claude Code for Scott to edit and approve
Scope: `articles/machine-learning/` in spolly74/azure-ai-docs, excluding `v1/` and `data-science-virtual-machine/`
Corpus profile: about 550 Markdown articles (409 root, 81 component-reference, 11 component-reference-v2, 46 prompt-flow, 3 known-issues), about 940K words, 95 include files, about 7,000 Markdown table rows

Each question lists the articles that should answer it, and the condition types it exercises (Goal 3). "Answer shape" says what a correct SPARQL result looks like.

## A. Compute

1. **Which compute targets can run training for automated ML, pipelines, and the designer?**
   Answer shape: compute target × workload, yes/no, with qualifiers like "SDK local mode only".
   Sources: `concept-compute-target.md`
   Conditions: interface (SDK, CLI, studio), release status

2. **Which compute targets support GPU for inference, and for which mode (real-time or batch)?**
   Sources: `concept-compute-target.md`, `concept-endpoints.md`

3. **What capabilities differ between a compute cluster and a compute instance?** (multinode, autoscale, idle shutdown, managed identity)
   Sources: `concept-compute-target.md`, `concept-compute-instance.md`, `how-to-create-attach-compute-cluster.md`

4. **What do I need to set up before attaching an Arc-enabled or AKS Kubernetes cluster, and in what order?**
   Answer shape: ordered prerequisite chain (extension → attach → instance types).
   Sources: `how-to-attach-kubernetes-anywhere.md`, `how-to-deploy-kubernetes-extension.md`, `how-to-attach-kubernetes-to-workspace.md`

5. **What are the limitations of serverless compute compared with a compute cluster?**
   Sources: `how-to-use-serverless-compute.md`

## B. Endpoints and deployment

6. **What are the differences between managed online endpoints, Kubernetes online endpoints, batch endpoints, and serverless API endpoints?**
   Sources: `concept-endpoints.md`, `concept-endpoints-online.md`, `concept-endpoints-batch.md`

7. **Which authentication methods does each endpoint type support (key, AML token, Microsoft Entra token)?**
   Sources: `concept-endpoints-online-auth.md`, `how-to-authenticate-online-endpoint.md`
   Conditions: endpoint type

8. **Which models in the model catalog can be deployed as serverless API endpoints, and in which regions?**
   Sources: `concept-endpoint-serverless-availability.md`
   Conditions: region

9. **What can a batch endpoint deploy (model, pipeline component), and what input data sources does it accept?**
   Sources: `concept-endpoints-batch.md`, `how-to-access-data-batch-endpoints-jobs.md`, `how-to-use-batch-model-deployments.md`

10. **Which steps and settings make up a safe (blue-green) rollout for an online endpoint?**
    Sources: `how-to-safely-rollout-online-endpoints.md`

11. **Which MLflow model flavors can be deployed without a scoring script, and to which targets?**
    Sources: `concept-mlflow-models.md`, `how-to-deploy-mlflow-models.md`
    Conditions: model flavor, deployment target

## C. Identity and access

12. **Which built-in roles exist for workspaces and registries, and what can each one do?**
    Sources: `how-to-assign-roles.md`

13. **Which role or permission is required to create compute, deploy to an endpoint, or create a registry?**
    Answer shape: action → minimum role or RBAC operation.
    Sources: `how-to-assign-roles.md`, `how-to-manage-registries.md`

14. **Which managed identities does a workspace use, and which Azure resources does each one need access to?**
    Sources: `how-to-identity-based-service-authentication.md`, `how-to-setup-authentication.md`

## D. Network security and data protection

15. **What outbound rules or service tags does a managed virtual network require in "allow internet outbound" versus "allow only approved outbound" mode?**
    Sources: `how-to-managed-network.md`, `concept-secure-network-traffic-flow.md`
    Conditions: isolation mode

16. **Which associated resources (Storage, Key Vault, Container Registry, Application Insights) need private endpoints for a fully isolated workspace?**
    Sources: `how-to-configure-network-isolation-with-v2.md`, `how-to-configure-private-link.md`, `how-to-network-security-overview.md`

17. **What data does Azure Machine Learning encrypt, and what changes when you use a customer-managed key?**
    Answer shape: data type → encryption method, with CMK effects (for example, extra resources created in a Microsoft-managed subscription).
    Sources: `concept-data-encryption.md`, `concept-customer-managed-keys.md`

18. **Which features are unsupported or limited when a workspace uses a customer-managed key or network isolation?**
    Sources: `concept-customer-managed-keys.md`, `how-to-managed-network.md`, `how-to-network-isolation-planning.md`
    Conditions: workspace configuration

19. **Which Azure Policy built-ins apply to Azure Machine Learning, and what does each one enforce?**
    Sources: `policy-reference.md`, `how-to-integrate-azure-policy.md`, `security-controls-policy.md`

## E. Data and features

20. **Which storage services can back a datastore, and which authentication methods does each support (credential-based or identity-based)?**
    Sources: `how-to-datastore.md`, `concept-data.md`, `how-to-administrate-data-authentication.md`

21. **What are the data asset types (uri_file, uri_folder, mltable), and when do you use each?**
    Sources: `concept-data.md`, `how-to-mltable.md`

22. **What are the managed feature store entities (feature store, entity, feature set, materialization), and how do they relate?**
    Sources: `concept-what-is-managed-feature-store.md`, `concept-top-level-entities-in-managed-feature-store.md`

## F. Training, AutoML, and pipelines

23. **Which task types does automated ML support, and which algorithms are available for each?**
    Sources: `concept-automated-ml.md`, `how-to-configure-auto-train.md`
    Conditions: task type, release status

24. **Which forecasting models and features (lags, calendar features, deep learning) does AutoML time-series forecasting use?**
    Sources: `concept-automl-forecasting-methods.md`, `concept-automl-forecasting-lags.md`, `concept-automl-forecasting-calendar-features.md`, `concept-automl-forecasting-deep-learning.md`

25. **Which job types can run as steps in a pipeline (command, sweep, parallel, Spark, AutoML)?**
    Sources: `concept-ml-pipelines.md`, `how-to-use-parallel-job-in-pipeline.md`, `how-to-use-sweep-in-pipeline.md`

26. **Which designer components exist, by category, and what inputs and outputs does each one have?**
    Sources: `component-reference-v2/`, `component-reference/`

## G. Assets, MLOps, and lifecycle

27. **Which asset types can be shared across workspaces through a registry, and which are workspace-only?**
    Sources: `concept-machine-learning-registries-mlops.md`, `how-to-share-models-pipelines-across-workspaces-with-registries.md`, `how-to-share-data-across-workspaces-with-registries.md`

28. **Which curated environments exist, and which frameworks and versions does each contain?**
    Sources: `resource-curated-environments.md`, `concept-environments.md`

29. **Which YAML schema defines each CLI v2 entity, and which fields are required?**
    Sources: `reference-yaml-*.md`
    Conditions: schema version

30. **What signals and metrics does model monitoring compute, and which ones need production data collection?**
    Sources: `concept-model-monitoring.md`, `concept-data-collection.md`, `how-to-monitor-model-performance.md`

## H. Limits, status, and change

31. **What are the service limits and default quotas for workspaces, runs, metrics, compute, and endpoints?**
    Answer shape: resource → limit → value, with "can be increased" yes/no.
    Sources: `resource-limits-capacity.md`, `how-to-manage-quotas.md`

32. **Which features are currently in preview?**
    Answer shape: feature list with release status taken from preview include files and "(preview)" headings.
    Conditions: release status

33. **Which features are deprecated or retiring, what replaces each one, and by what date?** (for example, prompt flow → Microsoft Agent Framework, data import → Microsoft Fabric)
    Sources: `prompt-flow/migrate-prompt-flow-to-agent-framework.md`, `data-import-migration-guide.md`, `azure-machine-learning-release-notes-cli-v2.md`
    Conditions: time

34. **Which known issues affect inferencing or Kubernetes compute, and what is the workaround for each?**
    Sources: `known-issues/`, `how-to-troubleshoot-online-endpoints.md`

35. **For each troubleshooting error message, what causes it and how do I fix it?**
    Answer shape: error → cause → resolution.
    Sources: `how-to-troubleshoot-*.md`

## Implied schema (input for `domain.yaml`, not a decision)

Classes these questions need: `ComputeTarget`, `Endpoint`, `Deployment`, `Model`, `ModelFlavor`, `Environment`, `DataAsset`, `Datastore`, `StorageService`, `Job`, `Pipeline`, `Component`, `Registry`, `Workspace`, `Role`, `Permission`, `ManagedIdentity`, `AzureResource`, `NetworkConfiguration`, `Policy`, `Region`, `Feature` (product capability), `Limit`, `KnownIssue`, `ErrorMessage`, `FeatureStoreEntity`, `Interface` (SDK, CLI, studio, REST).

Relationships: `supports`, `requires`, `runs_on`, `deploys_to`, `authenticates_with`, `grants`, `has_limit`, `available_in`, `replaced_by`, `incompatible_with`, `part_of`, `has_input` / `has_output`, `resolved_by`, `caused_by`.

Condition types beyond the current list: `region`, `release_status` (preview, GA, deprecated, retired), `interface`, `endpoint_type`, `workspace_configuration` (CMK, managed VNet mode), `date`.

## Notes from the content review

- Tables carry many of the facts (about 7,000 rows). Questions 1–3, 7, 12, 28 and 31 depend mostly on tables. Tier 3 extraction should get a table as a whole, never split across chunks.
- 104 articles use tabbed sections (`# [Python SDK](#tab/python)`), and many code blocks contain `#` lines. The chunker must not treat those as headings.
- Some in-scope articles link into `v1/`. Those links should be kept as references but not followed.
