using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using MLAgents;

using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

static class Extensions {
    public static void AddSafe(this Dictionary<int, Tuple<float, float>> dictionary, int key, Tuple<float, float> value) {
        if (!dictionary.ContainsKey(key))
            dictionary.Add(key, value);
    }
}

public class TurtleAcademy : Academy {
   
    public GameObject target;
    public Agent[] agents;
    
    private Maze mazeController = null;
    private Maze.Cell[] cells;
    public static GameObject wall;
    private static Dictionary<int, Tuple<float, float>> centerOfCells = new Dictionary<int, Tuple<float, float>>();
    private static int startingCell;
    private static int finalCell;
    private int totalCell = 9;
    private Tuple<float, float> xzPos;

    //3x3
    private float l = -2.365f;
    private float m = -0.035f;
    private float r = 2.295f;

    private int dir;

    private int counter = 0;    

    void Start() {
        fillCenters();
        wall = Resources.Load("Prefabs/MazeWall") as GameObject;
        mazeController = this.transform.gameObject.AddComponent<Maze>();
        xzPos = centerOfCells[UnityEngine.Random.Range(0, totalCell)];
        target.transform.position = new Vector3(xzPos.Item1, 0.05f, xzPos.Item2);
    }

    public override void AcademyReset() {
        xzPos = centerOfCells[UnityEngine.Random.Range(0, totalCell)];
        target.transform.position = new Vector3(xzPos.Item1, 0.05f, xzPos.Item2);
    }

    private void fillCenters() {
        centerOfCells.AddSafe(0, new Tuple<float, float>(l, l));
        centerOfCells.AddSafe(1, new Tuple<float, float>(m, l));
        centerOfCells.AddSafe(2, new Tuple<float, float>(r, l));

        centerOfCells.AddSafe(3, new Tuple<float, float>(l, m));
        centerOfCells.AddSafe(4, new Tuple<float, float>(m, m));
        centerOfCells.AddSafe(5, new Tuple<float, float>(r, m));
        
        centerOfCells.AddSafe(6, new Tuple<float, float>(l, r));
        centerOfCells.AddSafe(7, new Tuple<float, float>(m, r));
        centerOfCells.AddSafe(8, new Tuple<float, float>(r, r));
    }

    public class Maze : MonoBehaviour
    {
        [System.Serializable]
        public class Cell {
            public bool visited;
            public GameObject north;    //1
            public GameObject east;     //2
            public GameObject west;     //3
            public GameObject south;    //4
        }

        public float wallLength = 2.33f;
        public int xSize = 3;
        public int ySize = 3;

        private Vector3 initialPosition;
        private Vector3 targetPosition;

        public GameObject wallHolder;
        private int children;

        public Cell[] cells;

        private int currentCell = 0;
        private int totalCell = 9;

        private int visitedcells = 0;
        private bool startedBuild = false;
        private int currentNeighbour = 0;
        private List<int> lastCells;
        private int backingUp = 0;

        private int wallToBrake = 0;

        private GameObject t;

        public void Start() {
            UnityEngine.Random.seed = 0;
            createWalls();
        }

        public void createWalls() {
            wallHolder = new GameObject();
            wallHolder.name = "MazeHolder";

            initialPosition = new Vector3(-xSize/2 - wallLength/2 - 0.2f, 0.0f, -ySize/2 - 0.2f);
            Vector3 myPos = initialPosition;
            GameObject tmp;

            //xAxis
            for(int i=0; i<ySize; i++) {
                for(int j=0; j<=xSize; j++) {
                    myPos = new Vector3(initialPosition.x + (j*wallLength-wallLength/2), 0.0f, initialPosition.z + (i*wallLength - wallLength/2));
                    tmp = Instantiate(wall, myPos, Quaternion.Euler(0.0f, 0.0f, 0.0f)) as GameObject;
                    tmp.transform.parent = wallHolder.transform;
                }
            }

            //yAxis
            for(int i=0; i<=ySize; i++) {
                for(int j=0; j<xSize; j++) {
                    myPos = new Vector3(initialPosition.x + j*wallLength, 0.0f, initialPosition.z + i*wallLength - wallLength);
                    tmp = Instantiate(wall, myPos, Quaternion.Euler(0.0f, 90.0f, 0.0f)) as GameObject;
                    tmp.transform.parent = wallHolder.transform;
                }
            }

            createCells();
        }

        public void createCells() {
            lastCells = new List<int>();
            lastCells.Clear();
            children = wallHolder.transform.childCount;
            GameObject[] allWalls = new GameObject[children];
            int eastwestprocess = 0;
            int childprocess = 0;
            int termCount = 0;

            for(int i=0; i<children; i++)
                allWalls[i] = wallHolder.transform.GetChild(i).gameObject;

            cells = new Cell[xSize*ySize];
            for(int cellprocess=0; cellprocess<cells.Length; cellprocess++) {
                if(termCount == xSize) {
                    eastwestprocess++;
                    termCount = 0;
                }
                cells[cellprocess] = new Cell();
                cells[cellprocess].east = allWalls[eastwestprocess];
                cells[cellprocess].south = allWalls[childprocess+(xSize+1)*ySize];
                eastwestprocess++;
                termCount++;
                childprocess++;

                cells[cellprocess].west = allWalls[eastwestprocess];
                cells[cellprocess].north = allWalls[(childprocess+(xSize+1)*ySize)+xSize-1];
            }

            createMaze();
        }

        public void createMaze() {
            while(visitedcells < totalCell) {
                if(startedBuild) {
                    getNeighbour();
                    if(cells[currentNeighbour].visited == false && cells[currentCell].visited == true) {
                        brakeWall();
                        cells[currentNeighbour].visited = true;
                        visitedcells++;
                        lastCells.Add(currentCell);
                        currentCell = currentNeighbour;
                        if(lastCells.Count > 0)
                            backingUp = lastCells.Count - 1;
                    }
                } else {
                    //currentCell = UnityEngine.Random.Range(0, totalCell);
                    currentCell = 0;
                    cells[currentCell].visited = true;
                    visitedcells++;
                    startedBuild = true;
                }
            }
        }

        private void brakeWall() {
            switch(wallToBrake){
                case 1:
                    cells[currentCell].north.SetActive(false);
                    break;
                case 2:
                    cells[currentCell].east.SetActive(false);
                    break;
                case 3:
                    cells[currentCell].west.SetActive(false);
                    break;
                case 4:
                    cells[currentCell].south.SetActive(false);
                    break;
            }

        }

        private void getNeighbour() {
            int len = 0;
            int[] neighbours = new int[4];
            int[] connectingWall = new int[4];
            int check = (currentCell+1)/xSize;
            check -= 1;
            check *= xSize;
            check += xSize;

            if(currentCell+1 < totalCell && (currentCell+1) != check) {
                if(cells[currentCell + 1].visited == false) {
                    neighbours[len] = currentCell + 1;
                    connectingWall[len] = 3;
                    len++;
                }
            }
            
            if(currentCell-1 >= 0 && currentCell != check) {
                if(cells[currentCell - 1].visited == false) {
                    neighbours[len] = currentCell - 1;
                    connectingWall[len] = 2;
                    len++;
                }
            }
            
            if(currentCell + xSize < totalCell) {
                if(cells[currentCell + xSize].visited == false) {
                    neighbours[len] = currentCell + xSize;
                    connectingWall[len] = 1;
                    len++;
                }
            }
            
            if(currentCell - xSize >= 0) {
                if(cells[currentCell - xSize].visited == false) {
                    neighbours[len] = currentCell - xSize;
                    connectingWall[len] = 4;
                    len++;
                }
            }

            if(len > 0) {
                int x = UnityEngine.Random.Range(0, len);
                currentNeighbour = neighbours[x];
                wallToBrake = connectingWall[x];
            }
            else {
                if(backingUp > 0)
                    currentCell = lastCells[backingUp];
                    backingUp--;
            }
        }
    }
}