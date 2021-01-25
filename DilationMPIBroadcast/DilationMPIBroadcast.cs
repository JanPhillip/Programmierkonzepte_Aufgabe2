using System;
using OpenCvSharp;
using MPI;

namespace DilationMPIBroadcast
{
    /// <summary>
    /// Diese Klasse basiert auf dem Code-Beispiel aus der Vorlesung
    /// </summary>
    class DilationMPIBroadcast
    {
        /// <summary>
        /// Strukturelement mit den Maßen 5x5, das Sternförmig angeordnet ist
        /// </summary>
        public static byte[,] star5x5 = new byte[5, 5]{
            {0,0,1,0,0},
            {0,1,1,1,0},
            {1,1,1,1,1},
            {0,1,1,1,0},
            {0,0,1,0,0}
        };

        /// <summary>
        /// Strukturelement mit den Maßen 11x11
        /// </summary>
        public static byte[,] full11x11 = new byte[11, 11]{
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1}
        };

        /// <summary>
        /// Main-Programm mit MPI
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Pfad zum Bild
            string path = @"C:\Users\xovoo\Desktop\_temp\data\animal\1.kitten_small.jpg";

            // Integer-Array für Metainformationen des Bildes
            int[] imageProperties = new int[4];

            // Paralleler Code mithilfe von MPI
            MPI.Environment.Run(ref args, comm =>
            {
                var timer = System.Diagnostics.Stopwatch.StartNew();
                // Bild wird in Prozess 0 eingelesen und es werden Bildmaße, Type und Anzahl der Kanäle mit allen Prozessen geteilt
                #region Einlesen des Bildes und Verteilen der Bildinformationen
                Mat sourceMat = new Mat();

                // Bild wird in Prozess 0 eingelesen
                if (comm.Rank == 0)
                {
                    sourceMat = Cv2.ImRead(path, ImreadModes.AnyColor);
                    // Meta-Informationen zum Bild werden in das imageProperties-Array geschrieben, damit die Empfänger-Prozesse diese auch erhalten können
                    imageProperties = new int[4]
                    {
                        sourceMat.Rows,
                        sourceMat.Cols,
                        sourceMat.Type(),
                        sourceMat.Channels()
                    };
                    // Anzeigen des originalen Bildes
                    Cv2.ImShow("Original RGB", sourceMat);
                }
                comm.Barrier();
                // Verteilen der Meta-Daten an die anderen Prozesse
                comm.Broadcast(ref imageProperties, 0);
                comm.Barrier();
                #endregion

                // Das Bild wird streifenweise auf die Prozesse aufgeteilt, dort in YCbCr konvertiert und anschließend wieder in Prozess 0 zusammengefügt
                #region Farbraumkonvertierung RGB -> YCbCr
                
                // timer
                var timerYCbCr = System.Diagnostics.Stopwatch.StartNew();

                // reservieren des Speicherplatzes für die aufgeteilten Bilder, notwendig für Scatter()
                Mat splittedMat = new Mat(imageProperties[0] / comm.Size, imageProperties[1], imageProperties[2]);
                
                // Berechnung der Größe eines einzelnen Bildabschnitts
                int sendSize = imageProperties[0] / comm.Size * imageProperties[1] * imageProperties[3];

                // Verteilen des Bildes aus Prozess 0
                Unsafe.MPI_Scatter(sourceMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    splittedMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    0, Unsafe.MPI_COMM_WORLD);
                comm.Barrier();

                // Farbraumkonvertierung des Bild-Abschnitts
                splittedMat = RGB2YCbCr(splittedMat);
                comm.Barrier();

                // Zusammenfügen des Bildes in Prozess 0
                Unsafe.MPI_Gather(splittedMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    sourceMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    0, Unsafe.MPI_COMM_WORLD);
                comm.Barrier();
                timerYCbCr.Stop();
                Console.WriteLine("RGB to YCbCr: Process #" + comm.Rank + ": " + timerYCbCr.ElapsedMilliseconds + "ms");

                #endregion

                // Der Y-Kanal des Bildes wird per MPI_Bcast an alle Prozesse gesendet. Dort wird abschnittsweise die Dilatation angewendet. Anschließende wird das Bild in Prozess 0 wieder zusammengefügt
                #region Dilatation

                // Timer
                var timerDilatation = System.Diagnostics.Stopwatch.StartNew();

                // reservieren des Speichers in jedem Prozess für den Y-Kanal des Bildes, notwendig für MPI_Bcast
                Mat[] sourceChannel = new Mat[3];
                sourceChannel[0] = new Mat(imageProperties[0], imageProperties[1], MatType.CV_8UC1);

                // Spaltung des Bildes in einzelne Kanäle in Prozess 0
                if (comm.Rank == 0)
                {
                    sourceChannel = sourceMat.Split();
                }
                comm.Barrier();

                // Verteilen des Y-Kanals an alle Prozesse
                Unsafe.MPI_Bcast(sourceChannel[0].Data, imageProperties[0] * imageProperties[1], Unsafe.MPI_UNSIGNED_CHAR, 0, Unsafe.MPI_COMM_WORLD);

                // Belegter Speicher in der Größe des zu erwartenden Ausgangs-Bereichs aus der Dilatation
                Mat imageBlock = Mat.Zeros(imageProperties[0] / comm.Size, sourceChannel[0].Cols, MatType.CV_8UC1);

                // Morphologische Dilatation, Rückgabewert ist ein n-tel des Bildes (n=comm.Size), auf welches die Dilatation angewendet wurde
                imageBlock = Dilate(sourceChannel[0], star5x5, comm.Size, comm.Rank);
                //imageBlock = Dilate(sourceChannel[0], full11x11, comm.Size, comm.Rank);

                // Anzeigen / Speichern von Zwischenergebnissen
                //Cv2.ImWrite("Part" + comm.Rank + ".jpg", imageBlock);
                //Cv2.ImShow("Splitted Image: Rank #" + comm.Rank, imageBlock);

                comm.Barrier();

                // Zusammentragen der einzelnen Y-Bildabschnitte in ein Bild in Prozess 0
                Unsafe.MPI_Gather(imageBlock.Data, imageProperties[0] / comm.Size * imageProperties[1], Unsafe.MPI_UNSIGNED_CHAR,
                    sourceChannel[0].Data, imageProperties[0] / comm.Size * imageProperties[1] , Unsafe.MPI_UNSIGNED_CHAR,
                    0, Unsafe.MPI_COMM_WORLD);
                comm.Barrier();

                // Zusammenfügen der Bildkanäle in ein YCbCr-Bild (Prozess 0)
                if (comm.Rank == 0)
                {
                    Cv2.Merge(sourceChannel, sourceMat);
                }

                timerDilatation.Stop();
                Console.WriteLine("Dilatation: Process #" + comm.Rank + ": " + timerDilatation.ElapsedMilliseconds + "ms");
                #endregion

                // Das Bild wird streifenweise auf die Prozesse aufgeteilt, wo es zurück in den RGB-Farbraum konvertiert wird. Anschließend wird das Bild in Prozess 0 wieder zusammengefügt.
                #region Farbraumkonvertierung YCbCr -> RGB

                // Aufteilen des Bildes in Streifen auf alle Prozesse
                Unsafe.MPI_Scatter(sourceMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    splittedMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    0, Unsafe.MPI_COMM_WORLD);
                comm.Barrier();

                // Farbraumkonvertierung zu RGB des Bildabschnitts
                splittedMat = YCbCr2RGB(splittedMat);
                comm.Barrier();

                // Zusammenfügen der Bildabschnitte zu einem RGB-Bild in Prozess 0
                Unsafe.MPI_Gather(splittedMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    sourceMat.Data, sendSize, Unsafe.MPI_UNSIGNED_CHAR,
                    0, Unsafe.MPI_COMM_WORLD);
                comm.Barrier();
                #endregion

                // Das Bild wird dem Anwender per OpenCV angezeigt
                #region Anzeigen des Bildes
                timer.Stop();

                if (comm.Rank == 0)
                {
                    Console.WriteLine("Gesamtzeit (MPI): " + timer.ElapsedMilliseconds + "ms");
                    Cv2.ImShow("Final Image", sourceMat);
                    Cv2.ImWrite("out.jpg", sourceMat);
                }

                // Warten, bis eine Taste gedrückt wurde
                Cv2.WaitKey();
                Cv2.DestroyAllWindows();
                #endregion

            });
        }

        /// <summary>
        /// Converting a given RGB Mat to YCbCr
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static Mat RGB2YCbCr(Mat input)
        {
            if (input.Channels() < 3)
            {
                throw new Exception("Not enaugth channels");
            }

            Mat output = Mat.Zeros(input.Size(), MatType.CV_8UC3);

            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < input.Cols; j++)
                {

                    Vec3b inPixel = input.At<Vec3b>(i, j);
                    Vec3b outPixel = output.At<Vec3b>(i, j);

                    float r = inPixel.Item0;
                    float g = inPixel.Item1;
                    float b = inPixel.Item2;

                    float Y = 0.2989f * r + 0.5866f * g + 0.1145f * b;
                    float Cr = (r - Y) * 0.713f + 128f;
                    float Cb = (b - Y) * 0.564f + 128f;

                    outPixel[0] = Convert.ToByte(Y);
                    outPixel[1] = Convert.ToByte(Cr);
                    outPixel[2] = Convert.ToByte(Cb);

                    output.Set(i, j, outPixel);
                }
            }
            return output;
        }

        /// <summary>
        /// Converting a given YCbCr Mat to RGB
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static Mat YCbCr2RGB(Mat input)
        {
            if (input.Channels() < 3)
            {
                throw new Exception("Not enaugth channels");
            }

            Mat output = Mat.Zeros(input.Size(), MatType.CV_8UC3);

            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < input.Cols; j++)
                {
                    Vec3b inPixel = input.At<Vec3b>(i, j);
                    Vec3b outPixel = output.At<Vec3b>(i, j);

                    float Y = inPixel.Item0;
                    float Cr = inPixel.Item1;
                    float Cb = inPixel.Item2;

                    float r = MathF.Max(0, MathF.Min(255, (float)(Y + 1.403f * (Cr - 128))));
                    float g = MathF.Max(0, MathF.Min(255, (float)(Y - 0.3456f * (Cb - 128) - 0.7145f * (Cr - 128))));
                    float b = MathF.Max(0, MathF.Min(255, (float)(Y + 1.773f * (Cb - 128))));

                    outPixel[0] = Convert.ToByte(r);
                    outPixel[1] = Convert.ToByte(g);
                    outPixel[2] = Convert.ToByte(b);
                    output.Set(i, j, outPixel);
                }
            }
            return output;
        }

        /// <summary>
        /// Dilation algorithm
        /// </summary>
        /// <param name="input"></param>
        /// <param name="byteArray"></param>
        /// <param name="commSize"></param>
        /// <param name="commRank"></param>
        /// <returns></returns>
        public static Mat Dilate(Mat input, byte[,] byteArray, int commSize, int commRank)
        {
            Mat output = Mat.Zeros(input.Rows / commSize, input.Cols, MatType.CV_8UC1);

            // Size of the byteArray
            int xSize = byteArray.GetLength(0);
            int ySize = byteArray.GetLength(1);
            int startRow = commRank * (input.Rows / commSize);
            int endRow = startRow + (input.Rows / commSize);
            int outX = 0;
            int outY = 0;

            // Iterating over the image
            for (int i = startRow; i < endRow; i++)
            {
                for (int j = 0; j < input.Cols; j++)
                {
                    // Values for each pixel
                    byte max = 0;
                    byte currentValue = 0;

                    // Iterating over the byteArray
                    for (int x = 0; x < xSize; x++)
                    {
                        for (int y = 0; y < ySize; y++)
                        {
                            int posX = i + (x - xSize / 2);
                            int posY = j + (y - ySize / 2);
                            if (byteArray[x, y] > 0)
                            {
                                // Check for the bounds of the image
                                if (posX >= 0 && posY >= 0 && posX < input.Rows && posY < input.Cols)
                                {
                                    currentValue = input.At<byte>(posX, posY);
                                    if (currentValue > max)
                                    {
                                        max = currentValue;
                                    }
                                }
                            }
                        }
                    }
                    output.Set<byte>(outX, outY, max);
                    outY++;
                }
                outX++;
                outY = 0;
            }
            return output;
        }
    }
}
