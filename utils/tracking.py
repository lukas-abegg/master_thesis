from comet_ml import Experiment


def load_tracking(hyper_params):
    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                            project_name="project",
                            workspace="workspace")

    experiment.log_parameters(hyper_params)
    experiment.display()
    return experiment


def stop_tracking(experiment: Experiment):
    if experiment is not None:
        experiment.end()
