// mlagents-learn config/trainer_config.yaml --run-id=0 --train

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

public class TurtleAgent_2 : Agent {
    public int whichMazeAmI;

    public GameObject target;
    public Transform lidar;

    private Vector3 basePosition;
    private static Quaternion baseRotation;
    private static Vector3 center;

    private float[] rayCastDistances;
    private bool obstacleCollision = false;

    public int angular_0 = 0, angular_1 = 45, angular_2 = 90;
    public float linear_0 = 0.05f, linear_1 = 0.1f, linear_2 = 0.2f, linear_3 = 0.15f, linear_4 = 0.2f;

    private int scanNumber = 20;
    private float oldDistance, angleToTarget;

    private Maze mazeController = null;
    private Maze.Cell[] cells;
    public static GameObject wall;
    private static Dictionary<int, Tuple<float, float>> centerOfCells = new Dictionary<int, Tuple<float, float>>();
    private static int startingCell;
    private static int finalCell;

    //3x3
    private float l = -2.365f;
    private float m = -0.035f;
    private float r = 2.295f;

    private int dir;

    private int counter = 0;

    Rigidbody body;

    void Start() {
        fillCenters();
        basePosition = this.transform.position;
        baseRotation = this.transform.rotation;
        
        wall = Resources.Load("Prefabs/MazeWall") as GameObject;

        body = GetComponent<Rigidbody>();

        setMaze();

        rayCastDistances = new float[scanNumber];
        laserScan();

        setBotTarget();
    }

    public override void AgentReset() {
        setBotTarget();
        oldDistance = Vector3.Distance(this.transform.position, target.transform.position);
    }

    public override void CollectObservations() {
        // normalizedValue = (currentValue - minValue) / (maxValue - minValue)

        // Add lidar values (20)
        AddVectorObs(rayCastDistances); // Normalized in Update()

        // Add distance (1)
        float distanceToTarget = Vector3.Distance(this.transform.position, target.transform.position);
        AddVectorObs(distanceToTarget / 7f); // Normalization
        
        // Add Agent Orientation (1)
        Vector3 dir = target.transform.position - transform.position;
        angleToTarget = Vector3.SignedAngle(dir, transform.forward, Vector3.down);
        AddVectorObs(angleToTarget / 180f); // Normalization

    }


    public override void AgentAction(float[] vectorAction, string textAction) {
        // Default action
       
        switch (vectorAction[0]) {
            case 0:
                transform.Rotate (Vector3.up * angular_0 * Time.deltaTime);
                break;
            case 1:
                transform.Rotate (Vector3.down * angular_1 * Time.deltaTime);
                break;
            case 2:
                transform.Rotate (Vector3.down * angular_2 * Time.deltaTime);
                break;
            case 3:
                transform.Rotate (Vector3.up * angular_1 * Time.deltaTime);
                break;
            case 4:
                transform.Rotate (Vector3.up * angular_2 * Time.deltaTime);
                break;

            default:
                throw new ArgumentException("Invalid action value");   
        }

        /*
        switch (vectorAction[1]) {
            case 0:
                transform.Translate (Vector3.forward * linear_0 * Time.deltaTime);
                break;
            case 1:
                transform.Translate (Vector3.forward * linear_1 * Time.deltaTime);
                break;
            case 2:
                transform.Translate (Vector3.forward * linear_2 * Time.deltaTime);
                break;
            /*case 3:
                transform.Translate (Vector3.forward * linear_3 * Time.deltaTime);
                break;
            case 4:
                transform.Translate (Vector3.forward * linear_4 * Time.deltaTime);
                break;
            default:
                print(vectorAction[1]);
                throw new ArgumentException("Invalid action value");
        }
        */
        transform.Translate(Vector3.forward * linear_1 * Time.deltaTime);


        laserScan();
        
        // Reward and Done state
        float currentDistance = Vector3.Distance(this.transform.position, target.transform.position);

        float step = -0.0005f;

        float reward = 10f * (oldDistance - currentDistance);

        if (reward > 0 && vectorAction[0] == 0)
            reward = 15f * (oldDistance - currentDistance);

        if (currentDistance < 0.3f) {
            obstacleCollision = false;
            reward = 1f;
            Done();
        } else if(obstacleCollision) {
            obstacleCollision = false;
            reward = -1f;
            Done();
        } else if (this.transform.GetChild(0).transform.position.y < -0.3f) {
            reward = -1f;
            Done();
        } else {
            reward += step;
        }

        SetReward(reward);
        oldDistance = currentDistance;
    }

    void OnCollisionEnter(Collision collision) {
        if(collision.gameObject.tag == "Obstacle") {
            obstacleCollision = true;
            //print("Collided");
        }
    }

    private void laserScan() {
        // Draw Raycast
        RaycastHit hit;
        for (int i = 0; i < scanNumber; i++) {
            float angle = Mathf.Deg2Rad * (180 * (i / (float)(scanNumber-1)) - 90);
            Vector3 direction = new Vector3(Mathf.Sin(angle), 0, Mathf.Cos(angle));
            Physics.Raycast(lidar.transform.position, transform.TransformDirection(direction), out hit, 3.5f); // 3.5 is the plane length / 2
            Debug.DrawRay(lidar.transform.position, transform.TransformDirection(direction) * hit.distance, Color.red);

            rayCastDistances[i] = (hit.distance == 0) ? 3.5f : hit.distance;
            rayCastDistances[i] /= 3.5f; // Normalization
        }
    }

    public void setMaze() {
        mazeController = this.transform.gameObject.AddComponent<Maze>();
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

    public void setBotTarget() {
        center = this.transform.parent.position;

        setTurtleBot();
        setTarget();
    }

    private Tuple<float, float> getCoords(int cell) {
        return centerOfCells[cell];
    }

    private void setTurtleBot() {
        Tuple<float, float> tCoord = getCoords(0);

        this.transform.position = center + new Vector3(tCoord.Item1, 0.0f, tCoord.Item2);
    }

    private void setTarget() {
        UnityEngine.Random.seed = 0 + counter;
        counter++;
        int value = UnityEngine.Random.Range(1,16);

        Tuple<float, float> tCoord = getCoords(value);

        target.transform.position = center + new Vector3(tCoord.Item1, 0.05f, tCoord.Item2);
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
        private Vector3 pc;
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
            pc = this.transform.parent.position;
            createWalls();
        }

        public void createWalls() {
            wallHolder = new GameObject();
            wallHolder.name = "MazeHolder";

            //print("Maze center: " + pc);

            initialPosition = pc + new Vector3(-xSize/2 - wallLength/2 - 0.2f, 0.0f, -ySize/2 - 0.2f);
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