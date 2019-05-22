import plotter
import data_generator
import jidt_entropy_estimator
import telcs_entropy_estimator
import numpy as np




# this main funciton is full of ugly code leftovers, but this was the platform where I mainly ran my code

def main():
    num_observations = 1000
    num_dimension = 3
    num_runs = 50
    distribution = "gaussian_cube"

    jidt_entropy_estimator.init_jvm()
    results_gaussian = []
    # range_kernel = np.arange(0.03, 0.3, 0.02)
    # results_kernel = np.empty((num_runs, range_kernel.shape[0]))
    results_kozachenko = []
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
                normal_vector_series.append(0)

        results_gaussian += [jidt_entropy_estimator.entropy_calculator(time_series=time_series,
                                                                       num_dimension=num_dimension,
                                                                       type="gaussian")]
        # results_kernel[i] = [jidt_entropy_estimator.entropy_calculator(time_series=time_series,
        #                                                      num_dimension=num_dimension,
        #                                                      type="kernel",
        #                                                      kernel_width=r) for r in range_kernel]
        # results_kernel += [jidt_entropy_estimator.entropy_calculator(time_series=time_series,
        #                                                              num_dimension=num_dimension,
        #                                                              type="kernel",
        #                                                              kernel_width=r) for r in np.arange(0., 1., 0.05)]
        results_kozachenko += [jidt_entropy_estimator.entropy_calculator(time_series=time_series,
                                                                         num_dimension=num_dimension,
                                                                         type="kozachenko")]


    plotter.boxplotter(measures=[results_gaussian, results_kozachenko],
                       names=['gaussian', 'kozachenko'],
                       title=('Entropy estimate on '
                              + str(num_observations) + ' long time series in '
                              + str(num_dimension) + ' dimension ' +
                              'cube' + ' distribution'),
                       truth=data_generator.true_entopy_value(
                                 attractor_type=distribution,
                                 num_dimension=num_dimension))
    # print(results_kernel.shape)
    # print(results_kernel)
    # print(range_kernel.shape)
    # print(range_kernel)
    # plotter.boxplotter(measures=[results_kernel[:, r] for r in range(results_kernel.shape[1])],
    #                    names=[str(range_kernel[r])[0:4] for r in range(range_kernel.shape[0])],
    #                    title='kernel estimation with different r values on 3 dimensional uniform distribution',
    #                    truth=data_generator.true_entopy_value(
    #                              attractor_type=distribution,
    #                              num_dimension=num_dimension))



    # print('the average entropy for gaussian was %.3f with a deviation of %.4f'
    #       % (np.average(results_gaussian), np.std(results_gaussian)))
    # print('the average entropy for kernel was %.3f with a deviation of %.4f'
    #       % (np.average(results_kernel), np.std(results_kernel)))
    # print('the average entropy for kozachenko was %.3f with a deviation of %.4f'
    #       % (np.average(results_kozachenko), np.std(results_kozachenko)))

    # if num_dimension is 2:
    #     time_series_numpy = np.array(time_series)
    #     plotter.plot2d(time_series_numpy.T[0][:1000], time_series_numpy.T[1][:1000])
    #
    # if num_dimension is 3:
    #     time_series_numpy = np.array(time_series)
    #     plotter.plot3d(time_series_numpy.T[0], time_series_numpy.T[1],  time_series_numpy.T[2], normal_vector_series)
    #     plotter.plot2d(time_series_numpy.T[0][:1000], time_series_numpy.T[1][:1000])

if __name__ == "__main__":
    # execute only if run as a script
    main()