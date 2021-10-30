from sklearn.cluster import KMeans
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn_extra.cluster import KMedoids
import os
import docplex
from docplex.cp.model import CpoModel
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial import distance_matrix


N_RANGE = [2, 3, 4, 5, 6]
PRIORITIES = [1, 2, 3, 4, 5]
POPULATION_PATH = r"MopsiLocations2012-Joensuu.csv"
# LIB_OPTIMIZER = r'C:\Program Files\IBM\ILOG\CPLEX_Studio_Community201\cpoptimizer\bin\x64_win64\cpoptimizer.exe'
LIB_OPTIMIZER = r'/Applications/CPLEX_Studio201/cpoptimizer/bin/x86-64_osx/cpoptimizer'
HOSPITAL_PATH = r'hospitals.csv'
N_HOSPITALS = 10
ITERATIONS = 5
BASE_PLOTS_DIR = os.path.join("plots","pd-vdm")
OPTIMIZED_PLOTS_DIR = os.path.join("plots", "pd-vdm-o")


def plot_map(population, hospitals, priorities, save_to):
  fig, axes = plt.subplots(figsize=(10,10))
  axes.scatter(population[:,0], population[:,1], color="black",
               s=((priorities+3)*(priorities+3))*2, label="Population",
               alpha=0.6)
  axes.scatter(hospitals[:,0], hospitals[:,1], color="red", marker="X", label="Hospitals")
  plt.title("Population & Hospital Distribution")
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.legend()
  plt.savefig(os.path.join(save_to, "Map.png"))
  # plt.show()


def plot_distribution(priorities, save_to):
  fig, axes = plt.subplots(figsize=(10,10))
  distribution = [np.count_nonzero(priorities == priority) for priority in PRIORITIES]
  bar = axes.bar(PRIORITIES, distribution, align="center", color="dimgray")
  for rect in bar:
      height = rect.get_height()
      axes.text(rect.get_x() + rect.get_width() / 2, height + .5, str(height), ha="center", va="top")
  plt.title("Priority Distribution")
  plt.xlabel("Priority Level")
  plt.savefig(os.path.join(save_to, "Priorities.png"))
  # plt.show()


def print_solution(solution, DCs, staff, iterations):
  T = iterations
  for t in range(T):
    print(f"Iteration: {t}")
    for i in range(len(DCs)):
      print(f"\tHospital vaccinations {i}:")
      for j in range(staff[i]):
        # print(sum(staff_vaccinations))
        print("\t\t",solution[t][i][j])


def plot_solution(solution, DCs, population, staff, iterations, priorities, save_to):
  fig, axes = plt.subplots(figsize=(10,10))
  colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
  population = np.array(population)
  DCs = np.array(DCs)
  axes.scatter(population[:,0], population[:,1], color="black",
               s=((priorities+3)*(priorities+3))*2, label="Population",
               alpha=0.6)
  for index in range(len(DCs)):
    axes.scatter(np.array(DCs)[index,0], np.array(DCs)[index,1],
                 color=colors[index], marker="X")
  for t in range(iterations):
    for i in range(len(DCs)):
      for j in range(staff[i]):
        for k in range(len(population)):
          if solution[t][i][j][k]:
            axes.plot([DCs[i][0],population[k][0]], [DCs[i][1], population[k][1]], color=colors[i])
  plt.title("Vaccine Distribution")
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.savefig(os.path.join(save_to, "Solution.png"))
  # plt.show()


def get_clusters(clustering_class, data, save_to, range_n_clusters=None):
  if range_n_clusters is None:
      range_n_clusters = N_RANGE
  scores = []
  all_centers = []
  for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])
    clusterer = clustering_class(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    scores.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(data[:, 0], data[:, 1], marker='X', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                  fontsize=14, fontweight='bold')
    all_centers.append(centers)
    plt.savefig(os.path.join(save_to, "Cluster_{}.png".format(i)))
    plt.close()
  # plt.show()
  return scores, all_centers


def run_cplex(DCs, priorities, population, staff, vaccines, distances, iterations, alpha=1, beta=1, gamma=1):
    from docplex.cp.model import CpoModel
    T = iterations
    N = vaccines
    with CpoModel() as mdl:
        idx = [(i, j, k, t) for i in range(len(DCs)) for j in range(staff[i]) for k in range(len(population)) for t in
               range(T)]
        x = mdl.binary_var_dict(idx, name="x")
        mdl.add(
            mdl.sum(x[i, j, k, t] for k in range(len(population))) <= 1 for t in range(T) for i in range(len(DCs)) for j
            in range(staff[i]))
        mdl.add(mdl.sum(x[i, j, k, t] for t in range(T) for i in range(len(DCs)) for j in range(staff[i])) <= 1 for k in
                range(len(population)))
        mdl.add(mdl.sum(x[i, j, k, t] for t in range(T) for i in range(len(DCs)) for j in range(staff[i]) for k in
                        range(len(population))) <= N)
        mdl.add(mdl.maximize(mdl.sum(x[i, j, k, t] * (alpha + beta * priorities[k] - gamma * distances[k, i])
                                     for t in range(T) for i in range(len(DCs)) for j in range(staff[i]) for k in
                                     range(len(population)))))
        print("\nSolving model....")
        msol = mdl.solve(TimeLimit=10, execfile=LIB_OPTIMIZER)
        if msol:
            print("Solution status: " + msol.get_solve_status())
            return msol, x
    print("No solution found")
    return None, None


def copy_solution(msol, x, iterations, DCs, staff, population):
    solution = []
    for t in range(iterations):
        iteration = []
        for i in range(len(DCs)):
            hospital = []
            for j in range(staff[i]):
                staff_vaccines = []
                for k in range(len(population)):
                    staff_vaccines.append(msol[x[i, j, k, t]])
                hospital.append(staff_vaccines)
            iteration.append(hospital)
        solution.append(iteration)
    return solution


def evaluate_solutions(solution, iterations, DCs, staff, population):
    distances = []
    vaccinated = []
    for t in range(iterations):
        for i in range(len(DCs)):
            for j in range(staff[i]):
                for k in range(len(population)):
                    if solution[t][i][j][k]:
                        distances.append(
                            math.sqrt((DCs[i][0] - population[k][0]) ** 2 + (DCs[i][1] - population[k][1]) ** 2))
                        vaccinated.append(k)
    distances = np.array(distances)
    print("MIN: {}\nMAX: {}\nMEAN: {}\nSTD: {}\nTOTAL: {}".format(np.min(distances), np.max(distances),
                                                                  np.mean(distances), np.std(distances),
                                                                  np.sum(distances)))
    return vaccinated


def plot_vaccinations(vaccinated, population, priorities, save_to):
    priorities_vaccinated = [np.count_nonzero(priorities[vaccinated] == priority) for priority in PRIORITIES]
    unvaccinated = [index for index in range(len(population)) if index not in vaccinated]
    priorities_unvaccinated = [np.count_nonzero(priorities[unvaccinated] == priority) for priority in PRIORITIES]

    ax = plt.subplot(111)
    xticks = np.array(PRIORITIES)
    bar1 = ax.bar(xticks - 0.2, priorities_vaccinated, width=0.2, color='tab:orange', align='center',
                  label="Vaccinated")
    bar2 = ax.bar(xticks + 0.2, priorities_unvaccinated, width=0.2, color='tab:blue', align='center',
                  label="Unvaccinated")

    for rect in bar1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + .5, str(height), ha="center", va="top")
    for rect in bar2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + .5, str(height), ha="center", va="top")

    plt.legend()
    plt.title("Vaccines Distribution")
    plt.xlabel("Priority Level")
    plt.savefig(os.path.join(save_to, "Vaccinated.png"))
    # plt.show()


if __name__ == '__main__':
    population = pd.read_csv(POPULATION_PATH, header=None, delimiter=" ", dtype=float).to_numpy()[:50]
    hospitals = pd.read_csv(HOSPITAL_PATH, header=None, delimiter=" ", dtype=float).to_numpy()
    priorities = np.random.randint(low=min(PRIORITIES), high=max(PRIORITIES) + 1, size=len(population))
    staff = np.random.randint(low=1, high=2, size=len(hospitals))
    vaccines = np.random.randint(low=1, high=len(population))

    plot_map(population, hospitals, priorities, "plots")

    plot_distribution(priorities, "plots")

    # PD-VDM base
    scores, centers = get_clusters(KMedoids, hospitals, BASE_PLOTS_DIR)

    DCs_base = centers[np.argmax(scores)]
    distances_base = distance_matrix(population, DCs_base)
    print("Selected best {} hospitals with average silhouette score: {}".format(len(DCs_base), np.max(scores)))
    plot_map(population, DCs_base, priorities, BASE_PLOTS_DIR)

    msol_base, x_base = run_cplex(list(DCs_base), list(priorities), list(population), staff, vaccines, distances_base,
                                  ITERATIONS)
    solution_base = copy_solution(msol_base, x_base, ITERATIONS, DCs_base, staff, population)
    del msol_base, x_base

    print_solution(solution_base, DCs_base, staff, ITERATIONS)

    plot_solution(solution_base, DCs_base, population, staff, ITERATIONS, priorities, BASE_PLOTS_DIR)

    vaccinated = evaluate_solutions(solution_base, ITERATIONS, DCs_base, staff, population,)

    plot_vaccinations(vaccinated, population, priorities, BASE_PLOTS_DIR)

    # PD-VDM Optimized
    scores, centroids = get_clusters(KMeans, population, OPTIMIZED_PLOTS_DIR)

    centroids = centroids[np.argmax(scores)]
    distances_optimized = distance_matrix(centroids, hospitals)
    DCs_optimized = []
    for row in range(len(distances_optimized)):
        DCs_optimized.append(np.argmin(distances_optimized[row]))
    DCs_optimized = hospitals[DCs_optimized]
    distances_optimized = distance_matrix(population, DCs_optimized)
    plot_map(population, DCs_optimized, priorities, OPTIMIZED_PLOTS_DIR)

    msol_optimized, x_optimized = run_cplex(list(DCs_optimized), list(priorities), list(population), staff, vaccines,
                                            distances_optimized, ITERATIONS)
    solution_optimized = copy_solution(msol_optimized, x_optimized, ITERATIONS, DCs_optimized, staff, population)
    del msol_optimized, x_optimized

    print_solution(solution_optimized, DCs_optimized, staff, ITERATIONS)

    plot_solution(solution_optimized, DCs_optimized, population, staff, ITERATIONS, priorities, OPTIMIZED_PLOTS_DIR)

    vaccinated = evaluate_solutions(solution_optimized, ITERATIONS, DCs_optimized, staff, population)

    plot_vaccinations(vaccinated, population, priorities, OPTIMIZED_PLOTS_DIR)
