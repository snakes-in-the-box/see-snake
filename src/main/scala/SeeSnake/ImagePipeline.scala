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

  def pipeline(trainPath:String, testPath:String):(DataSetIterator, DataSetIterator) ={
    try {
      val testParentDir: File = new File(testPath)
      val testFilesInDir: FileSplit = new FileSplit(testParentDir, allowedExtensions)
      val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
      val pathFilter: BalancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker)
      val testFilesInDirSplit: Array[InputSplit] = testFilesInDir.sample(pathFilter, 100, 0)
      val testRecord: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
      testRecord.initialize(testFilesInDirSplit(0))
      val testData: DataSetIterator = new RecordReaderDataSetIterator(testRecord, 10, 1, outputNum)

      val trainParentDir: File = new File(trainPath)
      val trainFilesInDir: FileSplit = new FileSplit(trainParentDir, allowedExtensions)
      val trainFilesInDirSplit: Array[InputSplit] = trainFilesInDir.sample(pathFilter, 100, 0)
      val trainRecord: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
      trainRecord.initialize(trainFilesInDirSplit(0))
      val trainData: DataSetIterator = new RecordReaderDataSetIterator(trainRecord, 10, 1, outputNum)

      (trainData, testData)
    }catch {
      case ex:Exception => println(ex.toString)
        System.exit(-1)
        return null
    }
  }
}
