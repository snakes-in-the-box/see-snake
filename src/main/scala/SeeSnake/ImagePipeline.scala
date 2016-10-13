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

  def pipeline(path:String, train:Boolean):DataSetIterator ={
    try {
      if (!train) {
        val parentDir: File = new File(path)
        val filesInDir: FileSplit = new FileSplit(parentDir, allowedExtensions)
        val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
        val record: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
        record.initialize(filesInDir)
        val data: DataSetIterator = new RecordReaderDataSetIterator(record, 10, 1, outputNum)


        data
      }
      else {
        val parentDir: File = new File(path)
        val filesInDir: FileSplit = new FileSplit(parentDir, allowedExtensions)
        val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
        val pathFilter: BalancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker)
        val filesInDirSplit: Array[InputSplit] = filesInDir.sample(pathFilter, 100, 0)
        val record: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
        record.initialize(filesInDirSplit(0))
        val data: DataSetIterator = new RecordReaderDataSetIterator(record, 10, 1, outputNum)
        data
      }
    }catch {
      case ex:Exception => println(ex.toString)
        System.exit(-1)
        return null
    }
  }
}
