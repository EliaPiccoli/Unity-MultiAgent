// mlagents-learn config/trainer_config.yaml --run-id=0 --train

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using MLAgents;

using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

public class TurtleAgent_2 : Agent {
    public int AgentNumber;

    public Transform lidar;
    public GameObject target;

    private bool finishedEpisode;
    private Vector3 basePosition;
    private Quaternion baseRotation;
    private static Vector3 center;
    private Vector3 collisionPosition;

    private float[] rayCastDistances;
    private bool obstacleCollision = false;

    public int angular_0 = 0, angular_1 = 45, angular_2 = 90;
    public float linear_0 = 0.05f, linear_1 = 0.1f, linear_2 = 0.2f, linear_3 = 0.15f, linear_4 = 0.2f;

    private int scanNumber = 20;
    private float oldDistance, angleToTarget;

    private int counter = 0;

    Rigidbody body;

    void Start() {
        basePosition = this.transform.position;
        baseRotation = this.transform.rotation;

        body = GetComponent<Rigidbody>();

        rayCastDistances = new float[scanNumber];
        laserScan();

        setBot();
    }

    public override void AgentReset() {
        finishedEpisode = false;
        setBot();
        oldDistance = Vector3.Distance(this.transform.position, target.transform.position);
    }

    public override void CollectObservations() {
        // normalizedValue = (currentValue - minValue) / (maxValue - minValue)

        // Add lidar values (20)
        laserScan();                    // <-- We need to scan here, otherwise the first scan of each episode is wrong
        AddVectorObs(rayCastDistances); // Normalized in Update()

        // Add distance (1)
        float distanceToTarget = Vector3.Distance(this.transform.position, target.transform.position);
        AddVectorObs(distanceToTarget / 7f); // Normalization
        
        // Add Agent Orientation (1)
        Vector3 dir = target.transform.position - transform.position;
        angleToTarget = Vector3.SignedAngle(dir, transform.forward, Vector3.down);
        AddVectorObs(angleToTarget / 180f); // Normalization

        AddVectorObs(AgentNumber);

    }

    public bool EpisodeState() {
        return finishedEpisode;
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
            case 5:
                break;
            default:
                throw new ArgumentException("Invalid action value");   
        }

        if(vectorAction[0] != 5)
            transform.Translate(Vector3.forward * linear_1 * Time.deltaTime);
        else {
            this.transform.position = collisionPosition;
        }

        //laserScan();
        
        // Reward and Done state
        float currentDistance = Vector3.Distance(this.transform.position, target.transform.position);

        float step = -0.0005f;

        float reward = 10f * (oldDistance - currentDistance);

        if (reward > 0 && vectorAction[0] == 0)
            reward = 15f * (oldDistance - currentDistance);

        if (currentDistance < 0.3f) {
            obstacleCollision = false;
            reward = 1f;
            finishedEpisode = true;
            Done();
        } else if(obstacleCollision) {
            obstacleCollision = false;
            reward = -1f;
            finishedEpisode = true;
            Done();
        } else if (this.transform.GetChild(0).transform.position.y < -0.3f) {
            reward = -1f;
            finishedEpisode = true;
            Done();
        } else {
            reward += step;
        }

        SetReward(reward);
        oldDistance = currentDistance;
    }

    void OnCollisionEnter(Collision collision) {
        if(collision.gameObject.tag == "Obstacle" || collision.gameObject.tag == "Player") {
            obstacleCollision = true;
            collisionPosition = this.transform.position;
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

    public void setBot() {
        this.transform.position = basePosition;
        this.transform.rotation = baseRotation;
    }
}