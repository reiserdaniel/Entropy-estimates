from jpype import *
import os

def init_jvm():
    jarLocation = os.path.join(os.getcwd(), "..", "jidt.jar")
    if not(os.path.isfile(jarLocation)):
        exit("JIDT.jar not found (expected at "
             + os.path.abspath(jarLocation) + ") - are you running from demos/python?")
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


def entropy_calculator(TimeSeries, numDimension, type):
    TimeSeriesJava = JArray(JDouble, 2)(TimeSeries)
    if type is "gaussian":
        teCalcClass = JPackage("infodynamics.measures.continuous.gaussian").EntropyCalculatorMultiVariateGaussian
    elif type is "kernel":
        teCalcClass = JPackage("infodynamics.measures.continuous.kernel").EntropyCalculatorMultiVariateKernel
    elif type is "kozachenko":
        teCalcClass = JPackage("infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
    #elif type is "kraskov":
    #   teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").EntropyCalculatorMultiVariateKraskov
    teCalc = teCalcClass()
    teCalc.initialise(numDimension)
    teCalc.setObservations(TimeSeriesJava)
    return teCalc.computeAverageLocalOfObservations()
