name := "SeeSnake"
version := "0.1"
scalaVersion := "2.11.8"
classpathTypes += "maven-plugin"
resolvers ++= commonResolvers

libraryDependencies ++= Seq(
  "jfree" % "jfreechart" % "1.0.13",
  "commons-io" % "commons-io" % "2.4",
  "com.google.guava" % "guava" % "19.0",
  "jfree" % "jfreechart" % "1.0.13",
  "org.bytedeco" % "javacv" % "1.2",
  "org.datavec" % "datavec-data-codec" % "0.6.0",
  "org.deeplearning4j" % "arbiter-deeplearning4j" % "0.0.0.8",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "0.6.0",
  "org.deeplearning4j" % "deeplearning4j-ui" % "0.6.0",
  "org.jblas" % "jblas" % "1.2.4",
  "org.nd4j" % "nd4j-native-platform" % "0.6.0"
)

lazy val commonResolvers = Seq(
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/",
  "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"
)