import plotter
import data_generator
import jidt_entropy_estimator
import telcs_entropy_estimator
import numpy as np



def main():
    num_observations = 10000
    num_dimension = 3
    num_runs = 10
    distribution = "uniform"

    jidt_entropy_estimator.init_jvm()
    results_gaussian = []
    results_kernel = []
    results_kozachenko = []
    results_telcs_1 = []
    k_values_1 = []
    results_telcs_2 = []
    k_values_2 = []
    k_range = range(5, 50)
    time_series = []
    normal_vector_series = []

    for i in range(num_runs):
        time_series = []
        normal_vector_series = []
        if distribution is "simple_cube" or distribution is "gaussian_cube":
            for j in range(num_observations):
                vector, normal_vector = data_generator.new_time_data(distribution, num_dimension)
                time_series.append(vector)
                normal_vector_series.append(normal_vector)
        else:
            for j in range(num_observations):
                time_series.append(data_generator.new_time_data(distribution, num_dimension))

        results_gaussian += [jidt_entropy_estimator.entropy_calculator(time_series, num_dimension, "gaussian")]
        results_kernel += [jidt_entropy_estimator.entropy_calculator(time_series, num_dimension, "kernel")]
        results_kozachenko += [jidt_entropy_estimator.entropy_calculator(time_series, num_dimension, "kozachenko")]
        mean1, arr1 = telcs_entropy_estimator.entropy_estimate_v1(np.array(time_series), k_range)
        results_telcs_1 += [mean1]
        k_values_1 += [arr1]
        mean2, arr2 = telcs_entropy_estimator.entropy_estimate_v2(np.array(time_series), k_range)
        results_telcs_2 += [mean2]
        k_values_2 += [arr2]
    plotter.plot_k_range(k_range=k_range, k_values=np.array(k_values_1),
                         title='Entropy estimate for different k-s with algorithm 1 on ' \
                               + distribution + ' distribution')
    plotter.plot_k_range(k_range=k_range, k_values=np.array(k_values_2),
                         title='Entropy estimate for different k-s with algorithm 2 on ' \
                               + distribution + ' distribution')
    plotter.boxplotter(measures=[results_gaussian, results_kernel, results_kozachenko, results_telcs_1, results_telcs_2],
                       names=['gaussian', 'kernel', 'kozachenko', 'telcsv1', 'telcsv2'],
                       title=('Entropy estimate on '
                              + str(num_observations) + ' long time series in '
                              + str(num_dimension) + ' dimension ' +
                              distribution + ' distribution'))
                       # truth=np.log(np.sqrt(2*np.e*np.pi))*num_dimension)

    print('the average entropy for gaussian was %.3f with a deviation of %.4f'
          % (np.average(results_gaussian), np.std(results_gaussian)))
    print('the average entropy for kernel was %.3f with a deviation of %.4f'
          % (np.average(results_kernel), np.std(results_kernel)))
    print('the average entropy for kozachenko was %.3f with a deviation of %.4f'
          % (np.average(results_kozachenko), np.std(results_kozachenko)))

    if num_dimension is 2:
        time_series_numpy = np.array(time_series)
        plotter.plot2d(time_series_numpy.T[0][:1000], time_series_numpy.T[1][:1000])

    if num_dimension is 3:
        time_series_numpy = np.array(time_series)
        plotter.plot3d(time_series_numpy.T[0], time_series_numpy.T[1],  time_series_numpy.T[2], normal_vector_series)
        plotter.plot2d(time_series_numpy.T[0][:1000], time_series_numpy.T[1][:1000])

if __name__ == "__main__":
    # execute only if run as a script
    main()