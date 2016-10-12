package SeeSnake

import java.util.concurrent.TimeUnit

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxTimeIterationTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{LearningRatePolicy, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions


object Driver {

  def main(args: Array[String]) = {
    Nd4j.dtype = DataBuffer.Type.DOUBLE
    Nd4j.factory().setDType(DataBuffer.Type.DOUBLE)
    Nd4j.ENFORCE_NUMERICAL_STABILITY = true
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF)

    val nChannels = 3
    val outputNum = 10
    val batchSize = 128
    val nEpochs = 10
    val iterations = 1
    val seed = 12345
    val learnRate = .0001
    val dropOutRetainProbability = .9


    println("Build model....")

    val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(true).l2(0.0005)
      .learningRate(learnRate)
      .learningRateDecayPolicy(LearningRatePolicy.Sigmoid).lrPolicyDecayRate(0.1).lrPolicySteps(10)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS).momentum(0.5)
      .list()

      //TODO please add a .name() to each layer so we know what is going on where
      .layer(0, new ConvolutionLayer.Builder(3, 3)
      //nIn and nOut specify depth. nIn here is the nChannels and nOut is the # of filters to be applied
      .nIn(nChannels)
      .stride(1, 1)
      .padding(2, 2)
      .nOut(32)
      .activation("relu")
      .dropOut(dropOutRetainProbability)
      .build()
    )

      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build()
      )

      .layer(2, new ConvolutionLayer.Builder(3, 3)
        //Note that nIn needed be specified in later layers
        .stride(1, 1)
        .nOut(64)
        .activation("relu")
        .dropOut(dropOutRetainProbability)
        .build()
      )

      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build()
      )

      .layer(4, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .nOut(128)
        .activation("relu")
        .dropOut(dropOutRetainProbability)
        .build()
      )

      .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build()
      )

      .layer(6, new DenseLayer.Builder().activation("relu")
        .nOut(1024)
        .dropOut(dropOutRetainProbability)
        .build()
      )

      .layer(7, new DenseLayer.Builder().activation("relu")
        .nOut(512)
        .dropOut(dropOutRetainProbability)
        .build()
      )

      .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation("softmax")
        .build()
      )

      .backprop(true)
      .pretrain(false)

    new ConvolutionLayerSetup(builder, 32, 32, 3)

    val data = ImagePipeline.pipeline("/media/brad/disk2/cifar10/train")

    val conf: MultiLayerConfiguration = builder.build()

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()

    println("Train model....")
    //model.setListeners(new ScoreIterationListener(1))
    //model.setListeners(new HistogramIterationListener(1))
    (0 until nEpochs).foreach { i =>
      model.fit(data._1)
      println("********************************************************")
      println(s"*** Completed epoch ${i} with score ${model.score()}***")
      println("********************************************************")
    }

    println("Evaluate model....")
    val eval = new Evaluation(outputNum)
    while (data._2.hasNext) {
      val ds = data._2.next()
      val output = model.output(ds.getFeatureMatrix, false)
      eval.eval(ds.getLabels, output)
    }
    println(eval.stats())
    data._2.reset()

  }
}
