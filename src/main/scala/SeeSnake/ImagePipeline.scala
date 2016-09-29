package SeeSnake

import java.io.File
import java.util.Random

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{ImageTransform, MultiImageTransform, ShowImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator


object ImagePipeline {


  protected final val allowedExtensions: Array[String] = BaseImageLoader.ALLOWED_FORMATS

  protected final val seed: Long = 12345

  final val randNumGen: Random = new Random(seed)

  protected val height = 32
  protected val width = 32
  protected val channels = 3
  protected val outputNum = 10

  def mario(path:String):(DataSetIterator,DataSetIterator) ={

    val parentDir: File = new File(System.getProperty("user.dir"), path)
    val filesInDir: FileSplit = new FileSplit(parentDir, allowedExtensions, randNumGen)
    val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
    val pathFilter: BalancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker)
    val filesInDirSplit: Array[InputSplit] = filesInDir.sample(pathFilter, 80, 20)
    val trainData: InputSplit = filesInDirSplit(0)
    val testData: InputSplit = filesInDirSplit(1)
    val recordTest: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
    val recordTrain: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
    recordTest.initialize(testData)
    recordTrain.initialize(trainData)
    val dataTest: DataSetIterator = new RecordReaderDataSetIterator(recordTest, 10, 1, outputNum)
    val dataTrain: DataSetIterator = new RecordReaderDataSetIterator(recordTrain, 10, 1, outputNum)
    (dataTrain,dataTest)
  }
}
