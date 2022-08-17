using System.Diagnostics;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Linq;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.TaskbarClock;

namespace Seti
{
    internal class Seti
    {
        //private const String fileDirectory = @"G:\Seti\";
        //private const String fileDirectory = @"C:\Users\tordm\Documents\Visual Studio 2022\Projects\Seti\";
        private const String fileDirectory = @"C:\Users\tomal12\Visual Studio 2022\Projects\Seti\";

        internal static void Main()
        {
            DataSet train = new(true);
            train.Read();
            train.DisplayOne();
        }

        internal sealed class DataSet : Dictionary<Boolean, SortedList<String, DataRow>>
        {
            private readonly Boolean isTrain;
            private Double[,,] inputData;
            private Dictionary<Boolean, Double[,]> resonanceData;
            private Int32 workersLock;

            internal DataSet(Boolean isTrain)
            {
                this.isTrain = isTrain;
                Add(true, new SortedList<String, DataRow>());
                Add(false, new SortedList<String, DataRow>());
            }

            private void WorkerSelfSimilarity(Object parameters)
            {
                Int32 workerIndex = (Int32)((Object[])parameters)[0];
                Int32 frameNoise = (Int32)((Object[])parameters)[1];
                Double[,] resonanceTrue = new Double[273, 256];
                Double[,] resonanceFalse = new Double[273, 256];

                for (Int32 timeOffset = workerIndex; timeOffset < 273; timeOffset += Environment.ProcessorCount)
                {
                    for (Int32 frequencyOffset = 0; frequencyOffset < 256; frequencyOffset++)
                    {
                        MeanSpread meanSpreadTrue = new();
                        MeanSpread meanSpreadFalse = new();

                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            Int32 frequency2 = frequency + frequencyOffset;

                            if (frequency2 < 256)
                            {
                                for (Int32 time = 0; time < 273; time++)
                                {
                                    Int32 time2 = time + timeOffset;

                                    if (time2 < 273)
                                    {
                                        Double resonance = inputData[frameNoise, time, frequency] * inputData[frameNoise, time2, frequency2]
                                            /// (inputData[frameNoise, time, frequency] * inputData[frameNoise, time, frequency]
                                            //+ inputData[frameNoise, time2, frequency2] * inputData[frameNoise, time2, frequency2])
                                            ;

                                        if (!Double.IsNaN(resonance))
                                        {
                                            meanSpreadTrue.Add(resonance);
                                        }
                                    }

                                    time2 = time - timeOffset;

                                    if (time2 >= 0)
                                    {
                                        Double resonance = inputData[frameNoise, time, frequency] * inputData[frameNoise, time2, frequency2]
                                            /// (inputData[frameNoise, time, frequency] * inputData[frameNoise, time, frequency]
                                            //+ inputData[frameNoise, time2, frequency2] * inputData[frameNoise, time2, frequency2])
                                            ;

                                        if (!Double.IsNaN(resonance))
                                        {
                                            meanSpreadFalse.Add(resonance);
                                        }
                                    }
                                }
                            }
                        }

                        //Tord: change here!
                        resonanceTrue[timeOffset, frequencyOffset] = meanSpreadTrue.Mean;
                        resonanceFalse[timeOffset, frequencyOffset] = meanSpreadFalse.Mean;
                    }
                }

                do { Thread.Sleep(1); } while (Interlocked.CompareExchange(ref workersLock, 1, 0) == 1);
                for (Int32 timeOffset = workerIndex; timeOffset < 273; timeOffset += Environment.ProcessorCount)
                {
                    for (Int32 frequencyOffset = 0; frequencyOffset < 256; frequencyOffset++)
                    {
                        resonanceData[true][timeOffset, frequencyOffset] = resonanceTrue[timeOffset, frequencyOffset];
                        resonanceData[false][timeOffset, frequencyOffset] = resonanceFalse[timeOffset, frequencyOffset];
                    }
                }
                Interlocked.Exchange(ref workersLock, 0);
            }

            internal void DisplayOne()
            {
                Form2 form = new();

                if (Screen.AllScreens.Length > 1)
                {
                    //form.Location = new Point(-form.Width, 0);
                    form.Location = new Point(-form.Width, 273);
                    //form.Location = new Point(0, -form.Height - 273 * 2);
                }
                else
                {
                    form.Location = new Point(0, -32);
                }

                form.Show();
                Random random = new((Int32)(DateTime.Now.Ticks % Int32.MaxValue));
                String filename;

                do
                {
                    filename = this[true].ElementAt(random.Next(this[true].Count)).Key;
                }
                while (!File.Exists(fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy"));

                //filename = "9543918d5a7f353"; // debug clear
                //filename = "38e4f9f9620b680"; // super clear

                if (!File.Exists(fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy"))
                {
                    if (!Directory.Exists(fileDirectory + @"train\" + filename[..1]))
                    {
                        Directory.CreateDirectory(fileDirectory + @"train\" + filename[..1]);
                    }

                    File.Copy(@"D:\Seti\train\" + filename[..1] + @"\" + filename + ".npy",
                        fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy");
                }

                Single[] pythonData = Numpy.np.asfarray(Numpy.np.load(fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy"), Numpy.np.float32).GetData<Single>();

                #region Read and draw
                inputData = new Double[6, 273, 256];

                foreach (Int32 frame in new Int32[] { 0, 2, 4, 1, 3, 5 })
                {
                    for (Int32 time = 0; time < 273; time++)
                    {
                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            inputData[frame, time, frequency] = pythonData[(frame * 273 + time) * 256 + frequency];
                        }
                    }

                    List<Double> frameData = new();

                    for (Int32 frequency = 0; frequency < 256; frequency++)
                    {
                        for (Int32 time = 0; time < 273; time++)
                        {
                            frameData.Add(inputData[frame, time, frequency]);
                        }
                    }

                    frameData.Sort();

                    Double frameMinimum = frameData[0];

                    for (Int32 frequency = 0; frequency < 256; frequency++)
                    {
                        for (Int32 time = 0; time < 273; time++)
                        {
                            inputData[frame, time, frequency] -= frameMinimum;
                        }
                    }

                    Double frameMaximum = frameData[(Int32)(.95d * (Double)(frameData.Count))] - frameMinimum;
                    frameMinimum = frameData[(Int32)(.05d * (Double)(frameData.Count))] - frameMinimum;

                    Byte[] imageData = new Byte[3 * 273 * 256];

                    for (Int32 time = 0; time < 273; time++)
                    {
                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            Byte d;

                            if (inputData[frame, time, frequency] < frameMinimum)
                            {
                                d = 0;
                            }
                            else if (inputData[frame, time, frequency] > frameMaximum)
                            {
                                d = 255;
                            }
                            else
                            {
                                d = (Byte)((inputData[frame, time, frequency] - frameMinimum) / (frameMaximum - frameMinimum) * 255d);
                            }

                            Int32 index = 3 * (time * 256 + frequency);
                            imageData[index] = d;
                            imageData[index + 1] = d;
                            imageData[index + 2] = d;
                        }
                    }

                    Bitmap bitmap = new(256, 273, PixelFormat.Format24bppRgb);
                    BitmapData bitmapData = bitmap.LockBits(new Rectangle(new Point(0, 0), bitmap.Size), ImageLockMode.WriteOnly, bitmap.PixelFormat);
                    Marshal.Copy(imageData, 0, bitmapData.Scan0, 3 * bitmap.Height * bitmap.Width);
                    bitmap.UnlockBits(bitmapData);

                    switch (frame)
                    {
                        case 0: form.pictureBox0.Image = bitmap; form.pictureBox0.Update(); break;
                        case 1: form.pictureBox1.Image = bitmap; form.pictureBox1.Update(); break;
                        case 2: form.pictureBox2.Image = bitmap; form.pictureBox2.Update(); break;
                        case 3: form.pictureBox3.Image = bitmap; form.pictureBox3.Update(); break;
                        case 4: form.pictureBox4.Image = bitmap; form.pictureBox4.Update(); break;
                        case 5: form.pictureBox5.Image = bitmap; form.pictureBox5.Update(); break;
                        default: break;
                    }
                }
                #endregion

                { }

                foreach (Int32 frameNoise in new Int32[] { 1, 3, 5 })
                {
                    Fit fit = new((15 * 2 + 1) * (15 * 2 + 1) - 1, 1, true);
                    HashSet<(Int32 time, Int32 frequency)> offsets = new();

                    for (Int32 time = 0; time < 273; time++)
                    {
                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            List<Double> features = new();

                            for (Int32 timeOffset = -15; timeOffset <= 15; timeOffset++)
                            {
                                Int32 time2 = time + timeOffset;

                                if ((time2 >= 0) && (time2 < 273))
                                {
                                    for (Int32 frequencyOffset = -15; frequencyOffset <= 15; frequencyOffset++)
                                    {
                                        if (!(frequencyOffset.Equals(0) && timeOffset.Equals(0)))
                                        {
                                            Int32 frequency2 = frequency + frequencyOffset;

                                            if ((frequency2 >= 0) && (frequency2 < 256))
                                            {
                                                features.Add(inputData[frameNoise, time2, frequency2]);
                                            }
                                        }
                                    }
                                }
                            }

                            if (features.Count.Equals(960))
                            {
                                fit.Add(features.ToArray(), inputData[frameNoise, time, frequency]);
                            }
                        }
                    }

                    fit.Solve();

                    if (fit.SolutionExists)
                    {
                        Int32 i = 0;

                        for (Int32 timeOffset = -15; timeOffset <= 15; timeOffset++)
                        {
                            for (Int32 frequencyOffset = -15; frequencyOffset <= 15; frequencyOffset++)
                            {
                                if (!(frequencyOffset.Equals(0) && timeOffset.Equals(0)))
                                {
                                    Debug.WriteLine(timeOffset.ToString() + ";" + frequencyOffset.ToString() + ";" + fit.A[i].ToString());
                                    i++;
                                }
                            }
                        }
                    }
                    else
                    {
                        throw new Exception("Arse!");
                    }

                    { }

                    foreach (Int32 frame in new Int32[] { 0, 1, 2, 3, 4, 5 })
                    {
                        Double[,] recreatedMatrix = new Double[273, 256];

                        for (Int32 time = 0; time < 273; time++)
                        {
                            for (Int32 frequency = 0; frequency < 256; frequency++)
                            {
                                List<Double> features = new();

                                for (Int32 timeOffset = -15; timeOffset <= 15; timeOffset++)
                                {
                                    Int32 time2 = time + timeOffset;

                                    if ((time2 >= 0) && (time2 < 273))
                                    {
                                        for (Int32 frequencyOffset = -15; frequencyOffset <= 15; frequencyOffset++)
                                        {
                                            if (!(frequencyOffset.Equals(0) && timeOffset.Equals(0)))
                                            {
                                                Int32 frequency2 = frequency + frequencyOffset;

                                                if ((frequency2 >= 0) && (frequency2 < 256))
                                                {
                                                    features.Add(inputData[frame, time2, frequency2]);
                                                }
                                            }
                                        }
                                    }
                                }

                                if (features.Count.Equals(960))
                                {
                                    recreatedMatrix[time, frequency] = inputData[frame, time, frequency] - fit.Outcome(features.ToArray());
                                }
                            }
                        }

                        List<Double> recreatedMatrixValues = new();

                        for (Int32 time = 0; time < 273; time++)
                        {
                            for (Int32 frequency = 0; frequency < 256; frequency++)
                            {
                                recreatedMatrixValues.Add(recreatedMatrix[time, frequency]);
                            }
                        }

                        recreatedMatrixValues.Sort();

                        // DRAW
                        Bitmap bitmapMean = new(256, 273, PixelFormat.Format24bppRgb);

                        for (Int32 time = 0; time < 273; time++)
                        {
                            for (Int32 frequency = 0; frequency < 256; frequency++)
                            {
                                Double mean = recreatedMatrix[time, frequency];
                                Int32 searchIndex = recreatedMatrixValues.BinarySearch(mean);

                                if (searchIndex < 0)
                                {
                                }

                                Byte d = (Byte)((Double)searchIndex / recreatedMatrixValues.Count * 255d);
                                bitmapMean.SetPixel(frequency, time, Color.FromArgb(255, d, d, d));
                                inputData[frame, time, frequency] = recreatedMatrix[time, frequency];
                            }
                        }

                        switch (frame)
                        {
                            case 0: form.pictureBox0.Image = bitmapMean; form.pictureBox0.Update(); break;
                            case 1: form.pictureBox1.Image = bitmapMean; form.pictureBox1.Update(); break;
                            case 2: form.pictureBox2.Image = bitmapMean; form.pictureBox2.Update(); break;
                            case 3: form.pictureBox3.Image = bitmapMean; form.pictureBox3.Update(); break;
                            case 4: form.pictureBox4.Image = bitmapMean; form.pictureBox4.Update(); break;
                            case 5: form.pictureBox5.Image = bitmapMean; form.pictureBox5.Update(); break;
                            default: break;
                        }
                    }
                }


            }
            internal void DisplayOneSelfSimilarity()
            {
                Form2 form = new();

                if (Screen.AllScreens.Length > 1)
                {
                    //form.Location = new Point(-form.Width, 0);
                    form.Location = new Point(-form.Width, 273);
                    //form.Location = new Point(0, -form.Height - 273 * 2);
                }
                else
                {
                    form.Location = new Point(0, -32);
                }

                form.Show();
                Random random = new((Int32)(DateTime.Now.Ticks % Int32.MaxValue));
                String filename;

                do
                {
                    filename = this[true].OrderBy(_ => random.NextDouble()).First().Key;
                }
                while (!File.Exists(fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy"));

                //filename = "9543918d5a7f353"; // debug clear
                //filename = "38e4f9f9620b680"; // super clear

                if (!File.Exists(fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy"))
                {
                    if (!Directory.Exists(fileDirectory + @"train\" + filename[..1]))
                    {
                        Directory.CreateDirectory(fileDirectory + @"train\" + filename[..1]);
                    }

                    File.Copy(@"D:\Seti\train\" + filename[..1] + @"\" + filename + ".npy",
                        fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy");
                }

                Single[] pythonData = Numpy.np.asfarray(Numpy.np.load(fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy"), Numpy.np.float32).GetData<Single>();

                #region Read and draw
                inputData = new Double[6, 273, 256];

                foreach (Int32 frame in new Int32[] { 0, 2, 4, 1, 3, 5 })
                {
                    for (Int32 time = 0; time < 273; time++)
                    {
                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            inputData[frame, time, frequency] = pythonData[(frame * 273 + time) * 256 + frequency];
                        }
                    }

                    List<Double> frameData = new();

                    for (Int32 frequency = 0; frequency < 256; frequency++)
                    {
                        for (Int32 time = 0; time < 273; time++)
                        {
                            frameData.Add(inputData[frame, time, frequency]);
                        }
                    }

                    frameData.Sort();

                    Double frameMinimum = frameData[0];

                    for (Int32 frequency = 0; frequency < 256; frequency++)
                    {
                        for (Int32 time = 0; time < 273; time++)
                        {
                            inputData[frame, time, frequency] -= frameMinimum;
                        }
                    }

                    Double frameMaximum = frameData[(Int32)(.95d * (Double)(frameData.Count))] - frameMinimum;
                    frameMinimum = frameData[(Int32)(.05d * (Double)(frameData.Count))] - frameMinimum;

                    Byte[] imageData = new Byte[3 * 273 * 256];

                    for (Int32 time = 0; time < 273; time++)
                    {
                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            Byte d;

                            if (inputData[frame, time, frequency] < frameMinimum)
                            {
                                d = 0;
                            }
                            else if (inputData[frame, time, frequency] > frameMaximum)
                            {
                                d = 255;
                            }
                            else
                            {
                                d = (Byte)((inputData[frame, time, frequency] - frameMinimum) / (frameMaximum - frameMinimum) * 255d);
                            }

                            Int32 index = 3 * (time * 256 + frequency);
                            imageData[index] = d;
                            imageData[index + 1] = d;
                            imageData[index + 2] = d;
                        }
                    }

                    Bitmap bitmap = new(256, 273, PixelFormat.Format24bppRgb);
                    BitmapData bitmapData = bitmap.LockBits(new Rectangle(new Point(0, 0), bitmap.Size), ImageLockMode.WriteOnly, bitmap.PixelFormat);
                    Marshal.Copy(imageData, 0, bitmapData.Scan0, 3 * bitmap.Height * bitmap.Width);
                    bitmap.UnlockBits(bitmapData);

                    switch (frame)
                    {
                        case 0: form.pictureBox0.Image = bitmap; form.pictureBox0.Update(); break;
                        case 1: form.pictureBox1.Image = bitmap; form.pictureBox1.Update(); break;
                        case 2: form.pictureBox2.Image = bitmap; form.pictureBox2.Update(); break;
                        case 3: form.pictureBox3.Image = bitmap; form.pictureBox3.Update(); break;
                        case 4: form.pictureBox4.Image = bitmap; form.pictureBox4.Update(); break;
                        case 5: form.pictureBox5.Image = bitmap; form.pictureBox5.Update(); break;
                        default: break;
                    }
                }
                #endregion

                { }

                foreach (Int32 frameNoise in new Int32[] { 1, 3, 5 })
                {
                    resonanceData = new()
                    {
                        { true, new Double[273, 256] },
                        { false, new Double[273, 256] }
                    };



                    Thread[] workers = new Thread[Environment.ProcessorCount];
                    workersLock = 1;
                    for (Int32 workerIndex = 0; workerIndex < Environment.ProcessorCount; workerIndex++)
                    {
                        workers[workerIndex] = new Thread(WorkerSelfSimilarity) { Priority = ThreadPriority.BelowNormal, IsBackground = true };
                        workers[workerIndex].Start(new Object[] { workerIndex, frameNoise });
                    }
                    Interlocked.Exchange(ref workersLock, 0);
                    for (Int32 workerIndex = 0; workerIndex < Environment.ProcessorCount; workerIndex++)
                    {
                        workers[workerIndex].Join();
                    }
                    { }



                    List<Double> resonances = new();
                    List<Double> resonances2 = new();

                    for (Int32 timeOffset = 0; timeOffset < 273; timeOffset++)
                    {
                        for (Int32 frequencyOffset = 0; frequencyOffset < 256; frequencyOffset++)
                        {
                            if (!Double.IsNaN(resonanceData[true][timeOffset, frequencyOffset]))
                            {
                                resonances.Add(resonanceData[true][timeOffset, frequencyOffset]);
                            }

                            if (!Double.IsNaN(resonanceData[false][timeOffset, frequencyOffset]))
                            {
                                resonances2.Add(resonanceData[false][timeOffset, frequencyOffset]);
                            }
                        }
                    }

                    resonances.Sort();
                    resonances2.Sort();

                    Double resonanceMaximum = resonances.Reverse<Double>().Take(273 + 256).Reverse<Double>().First();
                    Double resonanceMaximum2 = resonances2.Reverse<Double>().Take(273 + 256).Reverse<Double>().First();

                    Byte[] imageData = new Byte[3 * 273 * 256];
                    Byte[] imageData2 = new Byte[3 * 273 * 256];
                    HashSet<(Int32 time, Int32 frequency)> offsets = new();

                    for (Int32 timeOffset = 0; timeOffset < 273; timeOffset++)
                    {
                        for (Int32 frequencyOffset = 0; frequencyOffset < 256; frequencyOffset++)
                        {
                            Int32 index = 3 * (timeOffset * 256 + frequencyOffset);

                            Int32 searchIndex = resonances.BinarySearch(resonanceData[true][timeOffset, frequencyOffset]);
                            Byte d1 = (Byte)((Double)searchIndex / resonances.Count * 255d);

                            searchIndex = resonances2.BinarySearch(resonanceData[false][timeOffset, frequencyOffset]);
                            Byte d2 = (Byte)((Double)searchIndex / resonances2.Count * 255d);

                            imageData[index] = 0;
                            imageData[index + 1] = resonanceData[true][timeOffset, frequencyOffset] > resonanceMaximum ? (Byte)0 : d1;
                            imageData[index + 2] = d1;

                            imageData2[index] = resonanceData[false][timeOffset, frequencyOffset] > resonanceMaximum2 ? (Byte)(d1 / 2 + d2 / 2) : (Byte)0;
                            imageData2[index + 1] = d2;
                            imageData2[index + 2] = d1;

                            if (resonanceData[true][timeOffset, frequencyOffset] > resonanceMaximum)
                            {
                                offsets.Add((timeOffset, frequencyOffset));
                            }

                            if (resonanceData[false][timeOffset, frequencyOffset] > resonanceMaximum2)
                            {
                                offsets.Add((timeOffset, frequencyOffset));
                            }
                        }
                    }

                    Bitmap bitmap = new(256, 273, PixelFormat.Format24bppRgb);
                    BitmapData bitmapData = bitmap.LockBits(new Rectangle(new Point(0, 0), bitmap.Size), ImageLockMode.WriteOnly, bitmap.PixelFormat);
                    Marshal.Copy(imageData, 0, bitmapData.Scan0, 3 * bitmap.Height * bitmap.Width);
                    bitmap.UnlockBits(bitmapData);
                    bitmap.RotateFlip(RotateFlipType.RotateNoneFlipY);

                    Bitmap bitmap2 = new(256, 273, PixelFormat.Format24bppRgb);
                    bitmapData = bitmap2.LockBits(new Rectangle(new Point(0, 0), bitmap2.Size), ImageLockMode.WriteOnly, bitmap2.PixelFormat);
                    Marshal.Copy(imageData2, 0, bitmapData.Scan0, 3 * bitmap2.Height * bitmap2.Width);
                    bitmap2.UnlockBits(bitmapData);
                    bitmap2.RotateFlip(RotateFlipType.RotateNoneFlipY);

                    switch (frameNoise)
                    {
                        case 1:
                            form.pictureBox1Transformed.Image = bitmap;
                            form.pictureBox1Transformed.Update();
                            form.pictureBox1Recreated.Image = bitmap2;
                            form.pictureBox1Recreated.Update();
                            break;
                        case 3:
                            form.pictureBox3Transformed.Image = bitmap;
                            form.pictureBox3Transformed.Update();
                            form.pictureBox3Recreated.Image = bitmap2;
                            form.pictureBox3Recreated.Update();
                            break;
                        case 5:
                            form.pictureBox5Transformed.Image = bitmap;
                            form.pictureBox5Transformed.Update();
                            form.pictureBox5Recreated.Image = bitmap2;
                            form.pictureBox5Recreated.Update();
                            break;
                        default: break;
                    }

                    { }

                    foreach (Int32 frame in new Int32[] { 0, 1, 2, 3, 4, 5 })
                    {
                        Double[,] recreatedMatrix = new Double[273, 256];

                        for (Int32 time = 0; time < 273; time++)
                        {
                            for (Int32 frequency = 0; frequency < 256; frequency++)
                            {
                                Double x = 0;
                                Double n = 0;

                                foreach (var offset in offsets)
                                {
                                    Int32 frequency2 = frequency + offset.frequency;

                                    if (frequency2 < 256)
                                    {
                                        Int32 time2 = time + offset.time;

                                        if (time2 < 273)
                                        {
                                            x += inputData[frame, time2, frequency2];
                                            n++;
                                        }

                                        time2 = time - offset.time;

                                        if (time2 >= 0)
                                        {
                                            x += inputData[frame, time2, frequency2];
                                            n++;
                                        }
                                    }
                                }

                                if (n > 0)
                                {
                                    if (x > 0)
                                    {
                                        recreatedMatrix[time, frequency] = inputData[frame, time, frequency] / (x / n);
                                    }
                                }
                            }
                        }

                        List<Double> recreatedMatrixValues = new();

                        for (Int32 time = 0; time < 273; time++)
                        {
                            for (Int32 frequency = 0; frequency < 256; frequency++)
                            {
                                recreatedMatrixValues.Add(recreatedMatrix[time, frequency]);
                            }
                        }

                        recreatedMatrixValues.Sort();

                        // DRAW
                        Bitmap bitmapMean = new(256, 273, PixelFormat.Format24bppRgb);

                        for (Int32 time = 0; time < 273; time++)
                        {
                            for (Int32 frequency = 0; frequency < 256; frequency++)
                            {
                                Double mean = recreatedMatrix[time, frequency];
                                Int32 searchIndex = recreatedMatrixValues.BinarySearch(mean);

                                if (searchIndex < 0)
                                {
                                }

                                Byte d = (Byte)((Double)searchIndex / recreatedMatrixValues.Count * 255d);
                                bitmapMean.SetPixel(frequency, time, Color.FromArgb(255, d, d, d));
                                inputData[frame, time, frequency] = recreatedMatrix[time, frequency];
                            }
                        }

                        switch (frame)
                        {
                            case 0: form.pictureBox0.Image = bitmapMean; form.pictureBox0.Update(); break;
                            case 1: form.pictureBox1.Image = bitmapMean; form.pictureBox1.Update(); break;
                            case 2: form.pictureBox2.Image = bitmapMean; form.pictureBox2.Update(); break;
                            case 3: form.pictureBox3.Image = bitmapMean; form.pictureBox3.Update(); break;
                            case 4: form.pictureBox4.Image = bitmapMean; form.pictureBox4.Update(); break;
                            case 5: form.pictureBox5.Image = bitmapMean; form.pictureBox5.Update(); break;
                            default: break;
                        }
                    }
                }


#warning fit with error weights, draw significance after
            }
            internal void Read()
            {
                if (isTrain)
                {
                    using StreamReader dataInput = new(new FileStream(fileDirectory + @"train_labels.csv", FileMode.Open));
                    String[] header = dataInput.ReadLine().Split(',');

                    while (dataInput.EndOfStream is false)
                    {
                        String[] dataStrings = dataInput.ReadLine().Split(',');

                        if (dataStrings[1].Equals("1"))
                        {
                            this[true].Add(dataStrings[0].Trim(), new DataRow());
                        }
                        else
                        {
                            this[false].Add(dataStrings[0].Trim(), new DataRow());
                        }
                    }

                    dataInput.Close();
                }
                else
                {
                    using StreamReader dataInput = new(new FileStream(fileDirectory + @"sample_submission.csv", FileMode.Open));
                    String[] header = dataInput.ReadLine().Split(',');

                    while (dataInput.EndOfStream is false)
                    {
                        String[] dataStrings = dataInput.ReadLine().Split(',');
                        this[false].Add(dataStrings[0].Trim(), new DataRow());
                    }

                    dataInput.Close();
                }
            }

            private sealed class MeanSpread
            {
                private Double sum;
                private Double sum2;

                internal Int32 count { get; private set; }

                internal Double Mean => sum / count;
                internal Double Spread => Math.Sqrt(Math.Abs(sum2 / count - Math.Pow(Mean, 2)));

                internal MeanSpread()
                {
                    sum = 0;
                    sum2 = 0;
                    count = 0;
                }

                internal void Add(Double resonance)
                {
                    sum += resonance;
                    sum2 += resonance * resonance;
                    count++;
                }
            }
        }

        internal sealed class DataRow
        {
            internal Double[,,] ImageData { get; private set; }

            internal DataRow()
            {
            }
        }

        private sealed class Fit
        {
            private readonly Int32 matrixSize;
            private readonly Double[] means;
            private readonly Double[] xTmp;
            private readonly Double[,] dispersionMatrix;
            private readonly Byte degree;
            private readonly Boolean useConstant;
            private Double weight;
            private readonly Int32 numberVariables;

            internal Double[] A { get; set; }
            internal Boolean SolutionExists { get; private set; }

            internal Fit(in Int32 numberVariables, in Byte degree, in Boolean useConstant)
            {
                this.numberVariables = numberVariables;
                this.degree = degree;
                this.useConstant = useConstant;

                if (useConstant)
                {
                    matrixSize = 1;
                }
                else
                {
                    matrixSize = 0;
                }

                matrixSize += numberVariables;

                if (this.degree > 1)
                {
                    matrixSize += numberVariables * (numberVariables + 1) / 2;
                }

                if (this.degree > 2)
                {
                    matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) / 6;
                }

                if (this.degree > 3)
                {
                    matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) * (numberVariables + 3) / 24;
                }

                if (this.degree > 4)
                {
                    matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) * (numberVariables + 3) * (numberVariables + 4) / 120;
                }

                if (this.degree > 5)
                {
                    matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) * (numberVariables + 3) * (numberVariables + 4) * (numberVariables + 5) / 720;
                }

                if ((matrixSize > 2048) | (matrixSize < 0))
                {
                    throw new NotImplementedException();
                }

                SolutionExists = false;
                A = null;
                xTmp = new Double[matrixSize];
                weight = 0;
                means = new Double[matrixSize];
                dispersionMatrix = new Double[matrixSize, matrixSize];
                //degreesFreedom = matrixSize;
            }

            private void Expand(in Double[] xVar)
            {
                Int32 idx = 0;

                if (useConstant)
                {
                    xTmp[idx++] = 1;
                }

                if (degree.Equals(1))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                    }
                }
                else if (degree.Equals(2))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                        }
                    }
                }
                else if (degree.Equals(3))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                            }
                        }
                    }
                }
                else if (degree.Equals(4))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                                for (Int32 i4 = i3; i4 < numberVariables; i4++)
                                {
                                    xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4];
                                }
                            }
                        }
                    }
                }
                else if (degree.Equals(5))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                                for (Int32 i4 = i3; i4 < numberVariables; i4++)
                                {
                                    xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4];
                                    for (Int32 i5 = i4; i5 < numberVariables; i5++)
                                    {
                                        xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4] * xVar[i5];
                                    }
                                }
                            }
                        }
                    }
                }
                else if (degree.Equals(6))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                                for (Int32 i4 = i3; i4 < numberVariables; i4++)
                                {
                                    xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4];
                                    for (Int32 i5 = i4; i5 < numberVariables; i5++)
                                    {
                                        xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4] * xVar[i5];
                                        for (Int32 i6 = i5; i6 < numberVariables; i6++)
                                        {
                                            xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4] * xVar[i5] * xVar[i6];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    throw new Exception("Degree = " + degree.ToString());
                }
            }

            internal void Add(in Double[] xVar, in Double target)
            {
                Expand(xVar);
                weight += 1d;

                for (Int32 i = 0; i < matrixSize; i++)
                {
                    means[i] += xTmp[i] * target;

                    for (Int32 j = i; j < matrixSize; j++)
                    {
                        dispersionMatrix[i, j] += xTmp[i] * xTmp[j];
                    }
                }
            }
            internal void Solve()
            {
                if ((weight <= 0)
                    || matrixSize.Equals(0))
                //|| (weight < degreesFreedom))
                {
                    A = null;
                    SolutionExists = false;
                    //MessageBox.Show(weight.ToString() + " < " + degreesFreedom.ToString());
                    return;
                }

                Double[,] mat = new Double[matrixSize, matrixSize];
                Double[] y = new Double[matrixSize];
                Boolean[] rowDone = new Boolean[matrixSize];
                Int32[] rowIndexes = new Int32[matrixSize];

                for (Int32 i = 0; i < matrixSize; i++)
                {
                    rowDone[i] = false;
                    rowIndexes[i] = -1;
                    y[i] = means[i] / weight;

                    for (Int32 j = i; j < matrixSize; j++)
                    {
                        mat[i, j] = dispersionMatrix[i, j] / weight;
                    }
                }

                for (Int32 j = 0; j < matrixSize; j++)
                {
                    for (Int32 k = j; k < matrixSize; k++)
                    {
                        mat[k, j] = mat[j, k];
                    }
                }

                // **** SOLVER ********
                Int32 currentColumn;
                Double maxValue;
                Int32 rowIndexMax;
                Int32 rowIndex;
                Double factor;
                Int32 columnIndex;
                Double[] tmpRow = new Double[matrixSize];
                Double tmpY;
                SolutionExists = true;

                for (currentColumn = 0; currentColumn < matrixSize; currentColumn++)
                {
                    maxValue = -1d;
                    rowIndexMax = -1;

                    for (rowIndex = 0; rowIndex < matrixSize; rowIndex++)
                    {
                        if (!rowDone[rowIndex])
                        {
                            if (Math.Abs(mat[currentColumn, rowIndex]) > maxValue)
                            {
                                rowIndexMax = rowIndex;
                                maxValue = Math.Abs(mat[currentColumn, rowIndexMax]);
                            }
                        }
                    }

                    if (rowIndexMax >= 0)
                    {
                        if (maxValue > Double.Epsilon)
                        {
                            factor = 1d / mat[currentColumn, rowIndexMax];

                            for (columnIndex = currentColumn; columnIndex < matrixSize; columnIndex++)
                            {
                                mat[columnIndex, rowIndexMax] *= factor;
                                tmpRow[columnIndex] = mat[columnIndex, rowIndexMax];
                            }

                            y[rowIndexMax] *= factor;
                            tmpY = y[rowIndexMax];
                            rowDone[rowIndexMax] = true;

                            for (rowIndex = 0; rowIndex < matrixSize; rowIndex++)
                            {
                                if (!rowDone[rowIndex])
                                {
                                    factor = mat[currentColumn, rowIndex];

                                    for (columnIndex = currentColumn; columnIndex < matrixSize; columnIndex++)
                                    {
                                        mat[columnIndex, rowIndex] -= tmpRow[columnIndex] * factor;
                                    }

                                    y[rowIndex] -= tmpY * factor;
                                }
                            }

                            rowIndexes[currentColumn] = rowIndexMax;
                        }
                        else
                        {
                            SolutionExists = false;
                            //MessageBox.Show(maxValue.ToString() + " <= " + Double.Epsilon.ToString());
                            break;
                        }
                    }
                    else
                    {
                        SolutionExists = false;
                        //MessageBox.Show(rowIndexMax.ToString() + " < 0");
                        break;
                    }
                }

                if (SolutionExists)
                {
                    Double tmpA;
                    A = new Double[matrixSize];

                    for (currentColumn = matrixSize - 1; currentColumn >= 0; currentColumn--)
                    {
                        rowIndex = rowIndexes[currentColumn];
                        tmpA = y[rowIndex];

                        for (columnIndex = matrixSize - 1; columnIndex > currentColumn; columnIndex--)
                        {
                            tmpA -= A[columnIndex] * mat[columnIndex, rowIndex];
                        }

                        A[currentColumn] = tmpA;
                    }
                }
            }
            internal Double Outcome(in Double[] xVar)
            {
                Expand(xVar);
                Double x = 0d;

                for (Int32 i = 0; i < matrixSize; i++)
                {
                    x += A[i] * xTmp[i];
                }

                return x;
            }
        }

    }
}