using System.Diagnostics;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Seti
{
    internal class Seti
    {
        //private const String fileDirectory = @"G:\Seti\";
        //private const String fileDirectory = @"C:\Users\tordm\Documents\Visual Studio 2022\Projects\Seti\";
        private const String fileDirectory = @"C:\Users\tomal12\Visual Studio 2022\Projects\Seti\";
        private static Int32 gridSize = 1;
        private static Int32 skipSize = 1;

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
            private Fit fit;
            private HashSet<(Int32 timeOffset, Int32 frequencyOffset)> offsets;

            private Int32 workersLock;

            internal DataSet(Boolean isTrain)
            {
                this.isTrain = isTrain;
                Add(true, new SortedList<String, DataRow>());
                Add(false, new SortedList<String, DataRow>());
            }

            private void WorkerFit(Object parameters)
            {
                Int32 workerIndex = (Int32)((Object[])parameters)[0];
                Int32 frameNoise = (Int32)((Object[])parameters)[1];
                Fit fitLocal = new(fit.numberVariables, fit.degree, fit.useConstant);
                HashSet<(Int32 timeOffset, Int32 frequencyOffset)> offsetsLocal = new();
                List<Double> features = new();
                Boolean offsetsSet = false;

                for (Int32 time = workerIndex + gridSize + skipSize; time < 273 - gridSize - skipSize; time += Environment.ProcessorCount)
                {
                    for (Int32 frequency = gridSize + skipSize; frequency < 256 - gridSize - skipSize; frequency++)
                    {
                        features.Clear();

                        if (!offsetsSet)
                        {
                            offsetsLocal.Clear();
                        }

                        for (Int32 timeOffset = -gridSize - skipSize; timeOffset <= gridSize + skipSize; timeOffset++)
                        {
                            Int32 time2 = time + timeOffset;
                            Boolean timeExclude = Math.Abs(timeOffset) <= skipSize;

                            if (time2 >= 0)
                            {
                                if (time2 < 273)
                                {
                                    for (Int32 frequencyOffset = -gridSize - skipSize; frequencyOffset <= gridSize + skipSize; frequencyOffset++)
                                    {
                                        Int32 frequency2 = frequency + frequencyOffset;
                                        Boolean frequencyExclude = Math.Abs(frequencyOffset) <= skipSize;

                                        if (frequency2 >= 0)
                                        {
                                            if (frequency2 < 256)
                                            {
                                                if (!(timeExclude && frequencyExclude))
                                                {
                                                    features.Add(inputData[frameNoise, time2, frequency2]);

                                                    if (!offsetsSet)
                                                    {
                                                        offsetsLocal.Add((timeOffset, frequencyOffset));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (features.Count.Equals(fit.numberVariables))
                        {
                            fitLocal.Add(features.ToArray(), inputData[frameNoise, time, frequency]);
                            offsetsSet = true;
                        }
                    }
                }

                do { Thread.Sleep(1); } while (Interlocked.CompareExchange(ref workersLock, 1, 0) == 1);
                fit.Add(fitLocal);

                if (offsets.Count.Equals(0))
                {
                    offsets = offsetsLocal.ToHashSet();
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

                String[] filenames = Directory.GetFiles(fileDirectory + @"train", "*.npy", SearchOption.AllDirectories).ToArray();

                Random random = new((Int32)(DateTime.Now.Ticks % Int32.MaxValue));
                String filename;

                do
                {
                    filename = this[true].ElementAt(random.Next(this[true].Count)).Key;
                }
                while (!File.Exists(fileDirectory + @"train\" + filename[..1] + @"\" + filename + ".npy"));

                filename = "9543918d5a7f353"; // debug clear

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
                Double[] means = new Double[6];
                Double[] spreads = new Double[6];

                foreach (Int32 frame in new Int32[] { 0, 2, 4, 1, 3, 5 })
                {
                    Double sum = 0;
                    Double sum2 = 0;
                    Int32 count = 0;
                    List<Double> frameData = new();

                    for (Int32 time = 0; time < 273; time++)
                    {
                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            Double x = pythonData[(frame * 273 + time) * 256 + frequency];
                            inputData[frame, time, frequency] = x;
                            frameData.Add(x);
                            sum += x;
                            sum2 += x * x;
                            count++;
                        }
                    }

                    means[frame] = sum / count;
                    spreads[frame] = Math.Sqrt(Math.Abs(sum2 / count - Math.Pow(means[frame], 2)));
                    frameData.Sort();

                    Byte[] imageData = new Byte[3 * 273 * 256];

                    for (Int32 time = 0; time < 273; time++)
                    {
                        for (Int32 frequency = 0; frequency < 256; frequency++)
                        {
                            Int32 searchIndex = frameData.BinarySearch(inputData[frame, time, frequency]);

                            if (searchIndex < 0)
                            {
                                throw new Exception("Weird");
                            }

                            Byte d = (Byte)((Double)searchIndex / frameData.Count * 255d);

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
                    fit = new(4 * gridSize * (gridSize + 2 * skipSize + 1), 1, true);
                    //fit = new((gridSize * 2 + 1) * (gridSize * 2 + 1) - 1, 1, true);
                    //fit = new(4 * gridSize, 1, true);

                    offsets = new();
                    Thread[] workers = new Thread[Environment.ProcessorCount];
                    workersLock = 1;

                    for (Int32 workerIndex = 0; workerIndex < Environment.ProcessorCount; workerIndex++)
                    {
                        workers[workerIndex] = new Thread(WorkerFit) { Priority = ThreadPriority.BelowNormal, IsBackground = true };
                        workers[workerIndex].Start(new Object[] { workerIndex, frameNoise });
                    }

                    Interlocked.Exchange(ref workersLock, 0);

                    for (Int32 workerIndex = 0; workerIndex < Environment.ProcessorCount; workerIndex++)
                    {
                        workers[workerIndex].Join();
                    }

                    fit.Solve();

                    if (fit.SolutionExists)
                    {
                        //Int32 i = 0;

                        //for (Int32 timeOffset = -gridSize; timeOffset <= gridSize; timeOffset++)
                        //{
                        //    if (timeOffset.Equals(0))
                        //    {
                        //        continue;
                        //    }

                        //    Debug.WriteLine("0;" + timeOffset.ToString() + ";" + fit.A[i++].ToString());
                        //}

                        //for (Int32 timeOffset = -gridSize; timeOffset <= gridSize; timeOffset++)
                        //{
                        //    if (timeOffset.Equals(0))
                        //    {
                        //        continue;
                        //    }

                        //    Debug.WriteLine(timeOffset.ToString() + ";0;" + fit.A[i++].ToString());
                        //}
                    }
                    else
                    {
                        throw new Exception("Arse!");
                    }

                    { }

                    foreach (Int32 frame in new Int32[] { 0, 1, 2, 3, 4, 5 })
                    {
                        // create noise matrix
                        // normalize to signal matrix
                        // subtract from signal matrix

                        Double[,] recreatedMatrix = new Double[273, 256];
                        Boolean[,] recreatedMatrixSet = new bool[273, 256];
                        Double sum = 0;
                        Double sum2 = 0;
                        Int32 count = 0;

                        for (Int32 time = gridSize + skipSize; time < 273 - gridSize - skipSize; time++)
                        {
                            for (Int32 frequency = gridSize + skipSize; frequency < 256 - gridSize - skipSize; frequency++)
                            {
                                List<Double> features = new();

                                foreach ((Int32 timeOffset, Int32 frequencyOffset) in offsets)
                                {
                                    Int32 time2 = time + timeOffset;

                                    if (time2 >= 0)
                                    {
                                        if (time2 < 273)
                                        {
                                            Int32 frequency2 = frequency + frequencyOffset;

                                            if (frequency2 >= 0)
                                            {
                                                if (frequency2 < 256)
                                                {
                                                    features.Add(inputData[frame, time2, frequency2]);
                                                }
                                            }
                                        }
                                    }
                                }

                                if (features.Count.Equals(fit.numberVariables))
                                {
                                    Double x = fit.Outcome(features.ToArray());
                                    recreatedMatrix[time, frequency] = x;
                                    recreatedMatrixSet[time, frequency] = true;
                                    sum += x;
                                    sum2 += x * x;
                                    count++;
                                }
                            }
                        }

                        Double mean = sum / count;
                        Double spread = Math.Sqrt(Math.Abs(sum2 / count - Math.Pow(mean, 2)));
                        List<Double> frameData = new();
                        sum = 0;
                        sum2 = 0;
                        count = 0;

                        if (frame.Equals(0))
                        {
                            StreamWriter debugOutput = new(new FileStream(fileDirectory + "debug.csv", FileMode.Create));

                            for (Int32 time = gridSize + skipSize; time < 273 - gridSize - skipSize; time++)
                            {
                                for (Int32 frequency = gridSize + skipSize; frequency < 256 - gridSize - skipSize; frequency++)
                                {
                                    if (recreatedMatrixSet[time, frequency])
                                    {
                                        debugOutput.WriteLine(frequency.ToString() + ";"
                                            + time.ToString() + ";"
                                            + inputData[frame, time, frequency].ToString() + ";"
                                            + recreatedMatrix[time, frequency].ToString());
                                    }
                                }
                            }

                            debugOutput.Close();
                        }

                        for (Int32 time = gridSize + skipSize; time < 273 - gridSize - skipSize; time++)
                        {
                            for (Int32 frequency = gridSize + skipSize; frequency < 256 - gridSize - skipSize; frequency++)
                            {
                                if (recreatedMatrixSet[time, frequency])
                                {
                                    inputData[frame, time, frequency] -= (recreatedMatrix[time, frequency] - mean) / spread * spreads[frame] + means[frame];
                                    frameData.Add(inputData[frame, time, frequency]);
                                    sum += inputData[frame, time, frequency];
                                    sum2 += inputData[frame, time, frequency] * inputData[frame, time, frequency];
                                    count++;
                                }
                            }
                        }

                        Debug.Write("Mean " + means[frame].ToString() + " -> ");
                        means[frame] = sum / count;
                        Debug.WriteLine(means[frame].ToString());
                        Debug.Write("Spread " + spreads[frame].ToString() + " -> ");
                        spreads[frame] = Math.Sqrt(Math.Abs(sum2 / count - Math.Pow(means[frame], 2)));
                        Debug.WriteLine(spreads[frame].ToString());
                        frameData.Sort();

                        // DRAW
                        Bitmap bitmapMean = new(256, 273, PixelFormat.Format24bppRgb);

                        for (Int32 time = gridSize + skipSize; time < 273 - gridSize - skipSize; time++)
                        {
                            for (Int32 frequency = gridSize + skipSize; frequency < 256 - gridSize - skipSize; frequency++)
                            {
                                if (recreatedMatrixSet[time, frequency])
                                {
                                    Int32 searchIndex = frameData.BinarySearch(inputData[frame, time, frequency]);

                                    if (searchIndex < 0)
                                    {
                                        throw new Exception("Weird");
                                    }

                                    Byte d = (Byte)((Double)searchIndex / frameData.Count * 255d);
                                    bitmapMean.SetPixel(frequency, time, Color.FromArgb(255, d, d, d));
                                }
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

                    skipSize *= 2;
                    gridSize *= 2;
                }


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

                internal Int32 Count { get; private set; }

                internal Double Mean => sum / Count;
                internal Double Spread => Math.Sqrt(Math.Abs(sum2 / Count - Math.Pow(Mean, 2)));

                internal MeanSpread()
                {
                    sum = 0;
                    sum2 = 0;
                    Count = 0;
                }

                internal void Add(Double resonance)
                {
                    sum += resonance;
                    sum2 += resonance * resonance;
                    Count++;
                }
            }
        }

        internal sealed class DataRow
        {
            internal Double[,,] ImageData { get; private set; }
            internal Double Mean { get; private set; }
            internal Double Spread { get; private set; }

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
            private Double weight;

            internal readonly Int32 numberVariables;
            internal readonly Byte degree;
            internal readonly Boolean useConstant;

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

            internal void Add(Fit fit)
            {
                weight += fit.weight;

                for (Int32 i = 0; i < matrixSize; i++)
                {
                    means[i] += fit.means[i];

                    for (Int32 j = i; j < matrixSize; j++)
                    {
                        dispersionMatrix[i, j] += fit.dispersionMatrix[i, j];
                    }
                }
            }
        }

    }
}