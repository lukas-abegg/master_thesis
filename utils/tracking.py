from comet_ml import Experiment


def load_tracking(params):
    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                            project_name=params["comet_project"],
                            workspace="abeggluk")

    experiment.log_parameters(params)
    experiment.display()
    return experiment


def stop_tracking(experiment: Experiment):
    if experiment is not None:
        experiment.end()
