package SeeSnake

import java.io.{File, FileInputStream, FileOutputStream}

import org.bytedeco.javacpp.opencv_shape.HistogramCostExtractor
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions


object Driver {

  def main(args: Array[String]) = {
    Nd4j.dtype = DataBuffer.Type.DOUBLE
    Nd4j.factory().setDType(DataBuffer.Type.DOUBLE)
    Nd4j.ENFORCE_NUMERICAL_STABILITY = true

    val nChannels = 3
    val outputNum = 10
    val batchSize = 64
    val nEpochs = 1
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
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS).momentum(0.1)
      .list()
      .layer(0, new ConvolutionLayer.Builder(3, 3)
        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the # of filters to be applied
        .nIn(nChannels)
        .stride(1, 1)
        .padding(2,2)
        .nOut(32)
        .activation("relu")
        .dropOut(dropOutRetainProbability)
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(2, new ConvolutionLayer.Builder(3, 3)
        //Note that nIn needed be specified in later layers
        .stride(1, 1)
        .nOut(64)
        .activation("relu")
        .dropOut(dropOutRetainProbability)
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(4, new ConvolutionLayer.Builder(3, 3)
        .stride(1,1)
        .nOut(128)
        .activation("relu")
        .dropOut(dropOutRetainProbability)
        .build())
      .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(6, new DenseLayer.Builder().activation("relu")
        .nOut(1024)
        .dropOut(dropOutRetainProbability).build())
      .layer(7, new DenseLayer.Builder().activation("relu")
        .nOut(512)
          .dropOut(dropOutRetainProbability).build())
      .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)

    // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
    new ConvolutionLayerSetup(builder, 32, 32, 3)

    val conf: MultiLayerConfiguration = builder.build()

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()


    val data = ImagePipeline.pipeline("C:/Users/Brent/Documents/School/DataPrac/cifar10/train/")

    /*
    println("Train model....")
    model.setListeners(new ScoreIterationListener(1))
    //model.setListeners(new HistogramIterationListener(1))
    (0 until nEpochs).foreach { i =>
      model.fit(data._1)
      println("*** Completed epoch {} ***", i)


      val modelFile = new File("C:/Users/Brent/Documents/School/DataPrac/model.bin")
      val fos = new FileOutputStream(modelFile)

      ModelSerializer.writeModel(model, fos, true)
    */

    println("Load model")
    val fis = new FileInputStream("C:/Users/Brent/Documents/School/DataPrac/model.bin");

    val network = ModelSerializer.restoreMultiLayerNetwork(fis);


    assert(network.getLayerWiseConfigurations().toJson() == model.getLayerWiseConfigurations().toJson());
    assert(network.params() == model.params());
    assert(network.getUpdater() == model.getUpdater());

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
    println("****************Example finished********************")
  //}

}
