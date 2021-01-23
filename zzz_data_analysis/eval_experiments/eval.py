from comet_ml.api import API
import pandas as pd

pd.options.display.float_format = '{:,.4f}'.format


def get_metric_data(experiment, metric):
    metrics = experiment.get_metrics(metric)
    hyp_1 = float(metrics[0]["metricValue"])
    hyp_2 = float(metrics[1]["metricValue"])
    return hyp_1, hyp_2


def build_hypothesis_data(experiment_name, beam, hypothesis, sari_val, bleu_val, sentence_bleu_val, f1_add_val,
                          f1_keep_val, p_del_val):
    return pd.Series(
        {"Experiment": experiment_name, "Beam": beam, "Hypothesis": hypothesis, "SARI": sari_val, "BLEU": bleu_val,
         "Sentence_BLEU": sentence_bleu_val, "F1_Add": f1_add_val, "F1_Keep": f1_keep_val,
         "P_Del": p_del_val, "Sum": sari_val + bleu_val})


def get_metrics(experiment, experiment_name):
    hypotheses_1 = []
    hypotheses_2 = []

    for i in [1, 2, 4, 6, 12]:
        sari_val_1, sari_val_2 = get_metric_data(experiment, "sari_score_" + str(i))
        bleu_val_1, bleu_val_2 = get_metric_data(experiment, "bleu_score_nltk_" + str(i))
        sentence_bleu_val_1, sentence_bleu_val_2 = get_metric_data(experiment, "avg_sentence_bleu_scores_" + str(i))
        f1_add_val_1, f1_add_val_2 = get_metric_data(experiment, "f1_add_" + str(i))
        f1_keep_val_1, f1_keep_val_2 = get_metric_data(experiment, "f1_keep_" + str(i))
        p_del_val_1, p_del_val_2 = get_metric_data(experiment, "p_del_" + str(i))

        hypotheses_1.append(build_hypothesis_data(experiment_name, i, 1, sari_val_1, bleu_val_1, sentence_bleu_val_1,
                                                  f1_add_val_1, f1_keep_val_1, p_del_val_1))
        hypotheses_2.append(build_hypothesis_data(experiment_name, i, 2, sari_val_2, bleu_val_2, sentence_bleu_val_2,
                                                  f1_add_val_2, f1_keep_val_2, p_del_val_2))

    return hypotheses_1, hypotheses_2


def prepare_project(experiment_name):
    experiments = comet_api.get_experiments(workspace, experiment_name)

    metric_data = []
    for i in experiments:
        experiment = comet_api.get_experiment_by_id(i.id)
        metric_data_hypotheses_1, metric_data_hypotheses_2 = get_metrics(experiment, experiment.name)
        metric_data.append(pd.DataFrame(metric_data_hypotheses_1))
        metric_data.append(pd.DataFrame(metric_data_hypotheses_2))

    overview_project = pd.concat(metric_data)

    return [overview_project, metric_data]


comet_api = API(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra")
workspace = "abeggluk"

projects_eval_names_mws = ["bart-mws-eval", "transformer-mws-eval-beam"]

projects_eval_mws = []
for i in projects_eval_names_mws:
    print(i)
    projects_eval_mws.append(prepare_project(i))

projects_eval_names_newsela = ["bart-newsela-eval", "transformer-newsela-eval-beam"]

projects_eval_newsela = []
for i in projects_eval_names_newsela:
    print(i)
    projects_eval_newsela.append(prepare_project(i))

best_beam_projects_eval_mws = pd.DataFrame([
    projects_eval_mws[0][1][0].iloc[3],
    projects_eval_mws[0][1][1].iloc[4],
    projects_eval_mws[0][1][2].iloc[4],
    projects_eval_mws[0][1][3].iloc[2],
    projects_eval_mws[0][1][4].iloc[3],
    projects_eval_mws[0][1][5].iloc[1],
    projects_eval_mws[0][1][6].iloc[4],
    projects_eval_mws[0][1][7].iloc[2],
    # ------------------------------- #
    projects_eval_mws[1][1][0].iloc[4],
    projects_eval_mws[1][1][1].iloc[4],
    projects_eval_mws[1][1][2].iloc[0],
    projects_eval_mws[1][1][3].iloc[4],
    projects_eval_mws[1][1][4].iloc[0],
    projects_eval_mws[1][1][5].iloc[1],
    projects_eval_mws[1][1][6].iloc[0],
    projects_eval_mws[1][1][7].iloc[1],
    projects_eval_mws[1][1][8].iloc[0],
    projects_eval_mws[1][1][9].iloc[1],
    projects_eval_mws[1][1][10].iloc[0],
    projects_eval_mws[1][1][11].iloc[1],
    projects_eval_mws[1][1][12].iloc[0],
    projects_eval_mws[1][1][13].iloc[1]
])

best_beam_projects_eval_newsela = pd.DataFrame([
    projects_eval_newsela[0][1][0].iloc[4],
    projects_eval_newsela[0][1][1].iloc[4],
    projects_eval_newsela[0][1][2].iloc[0],
    projects_eval_newsela[0][1][3].iloc[0],
    projects_eval_newsela[0][1][4].iloc[3],
    projects_eval_newsela[0][1][5].iloc[4],
    projects_eval_newsela[0][1][6].iloc[0],
    projects_eval_newsela[0][1][7].iloc[0],
    projects_eval_newsela[0][1][8].iloc[4],
    projects_eval_newsela[0][1][9].iloc[3],
    projects_eval_newsela[0][1][10].iloc[4],
    projects_eval_newsela[0][1][11].iloc[4],
    projects_eval_newsela[0][1][12].iloc[3],
    projects_eval_newsela[0][1][13].iloc[3],
    # ------------------------------- #
    projects_eval_newsela[1][1][0].iloc[4],
    projects_eval_newsela[1][1][1].iloc[1],
    projects_eval_newsela[1][1][2].iloc[0],
    projects_eval_newsela[1][1][3].iloc[1],
    projects_eval_newsela[1][1][4].iloc[0],
    projects_eval_newsela[1][1][5].iloc[1],
    projects_eval_newsela[1][1][6].iloc[0],
    projects_eval_newsela[1][1][7].iloc[1],
    projects_eval_newsela[1][1][8].iloc[0],
    projects_eval_newsela[1][1][9].iloc[1],
    projects_eval_newsela[1][1][10].iloc[0],
    projects_eval_newsela[1][1][11].iloc[1],
    projects_eval_newsela[1][1][12].iloc[0],
    projects_eval_newsela[1][1][13].iloc[1]
])

best_beam_projects_eval_mws = best_beam_projects_eval_mws.reset_index()

best_beam_projects_eval_mws_1 = []
best_beam_projects_eval_mws_2 = []
for i in range(len(best_beam_projects_eval_mws)):
    if i % 2 == 0:
        best_beam_projects_eval_mws_1.append(best_beam_projects_eval_mws.iloc[i])
    else:
        best_beam_projects_eval_mws_2.append(best_beam_projects_eval_mws.iloc[i])

best_beam_projects_eval_mws_1 = pd.DataFrame(best_beam_projects_eval_mws_1).reset_index()
best_beam_projects_eval_mws_2 = pd.DataFrame(best_beam_projects_eval_mws_2).reset_index()

best_beam_projects_eval_newsela = best_beam_projects_eval_newsela.reset_index()

best_beam_projects_eval_newsela_1 = []
best_beam_projects_eval_newsela_2 = []
for i in range(len(best_beam_projects_eval_newsela)):
    if i % 2 == 0:
        best_beam_projects_eval_newsela_1.append(best_beam_projects_eval_newsela.iloc[i])
    else:
        best_beam_projects_eval_newsela_2.append(best_beam_projects_eval_newsela.iloc[i])

best_beam_projects_eval_newsela_1 = pd.DataFrame(best_beam_projects_eval_newsela_1).reset_index()
best_beam_projects_eval_newsela_2 = pd.DataFrame(best_beam_projects_eval_newsela_2).reset_index()

eval_data_mws = []
for i in [3, 1, 2, 4, 7, 8, 10]:
    eval_data_mws.append(best_beam_projects_eval_mws_1.iloc[i])
    eval_data_mws.append(best_beam_projects_eval_mws_2.iloc[i])

eval_data_mws = pd.DataFrame(eval_data_mws)

eval_data_newsela = []
for i in [0, 1, 5, 6, 8, 9, 11, 13]:
    eval_data_newsela.append(best_beam_projects_eval_newsela_1.iloc[i])
    eval_data_newsela.append(best_beam_projects_eval_newsela_2.iloc[i])

eval_data_newsela = pd.DataFrame(eval_data_newsela)

print(eval_data_mws)
print(eval_data_newsela)
