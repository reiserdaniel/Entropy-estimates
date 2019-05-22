from jpype import *
import os

def init_jvm():
    jarLocation = os.path.join(os.getcwd(),  "infodynamics.jar")
    if not(os.path.isfile(jarLocation)):
        exit("JIDT.jar not found (expected at "
             + os.path.abspath(jarLocation) + ") - are you running from demos/python?")
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


def entropy_calculator(time_series, num_dimension, type, kernel_width=0.25):
    TimeSeriesJava = JArray(JDouble, 2)(time_series)
    if type is "gaussian":
        teCalcClass = JPackage("infodynamics.measures.continuous.gaussian").EntropyCalculatorMultiVariateGaussian
        teCalc = teCalcClass()
        teCalc.initialise(num_dimension)
    elif type is "kernel":
        teCalcClass = JPackage("infodynamics.measures.continuous.kernel").EntropyCalculatorMultiVariateKernel
        teCalc = teCalcClass()
        teCalc.setProperty("NORMALISE", "true")
        teCalc.initialise(num_dimension, kernel_width)
    elif type is "kozachenko":
        teCalcClass = JPackage("infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
        teCalc = teCalcClass()
        teCalc.initialise(num_dimension)

    teCalc.initialise(num_dimension)
    teCalc.setObservations(TimeSeriesJava)
    return teCalc.computeAverageLocalOfObservations()
