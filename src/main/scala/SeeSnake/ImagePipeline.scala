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


class ImagePipeline {


  protected final val allowedExtensions: Array[String] = BaseImageLoader.ALLOWED_FORMATS

  protected final val seed: Long = 12345

  final val randNumGen: Random = new Random(seed)

  protected val height = 50
  protected val width = 50
  protected val channels = 3
  protected val numExamples = 80
  protected val outputNum = 4

  def main(args: Array[String]) {

    val parentDir: File = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/DataExamples/ImagePipeline/")
    val filesInDir: FileSplit = new FileSplit(parentDir, allowedExtensions, randNumGen)
    val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
    val pathFilter: BalancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker)
    val filesInDirSplit: Array[InputSplit] = filesInDir.sample(pathFilter, 80, 20)
    val trainData: InputSplit = filesInDirSplit(0)
    val testData: InputSplit = filesInDirSplit(1)
    val recordReader: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)

    //val transform: ImageTransform = new MultiImageTransform(randNumGen, new ShowImageTransform("Display - before "))

    recordReader.initialize(trainData)
    var dataIter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum)
    while (dataIter.hasNext) {
      val ds = dataIter.next()
      println(ds)
      try {
        Thread.sleep(3000);
      } catch {
        case ex: InterruptedException =>
          Thread.currentThread().interrupt()
      }
    }
    recordReader.reset()

    recordReader.initialize(trainData)
    dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum)
    while (dataIter.hasNext) {
      val ds = dataIter.next()
    }
    recordReader.reset()

  }
}
