# Unity-MultiAgent
Set up to create and handle a multi agent environment using Unity ml-agents tool.

The project contains:
  1. Unity project of the environment: Maze_Multi_3x3
  2. Python files to interact with the Unity API using main.py
  3. The file agents.py contains the structure of the controller between the environment and the various AI.
  
This set up was made in order to create an enviroment with multiple agents; in this case the agents have to navigate the maze
and reach the red square.

This set up helps you to handle multiple agents which have their own episode but in order to go to the next one have to wait 
that all the other agents have finished. Since in ml-agents tool this it's not a straightfoward feature using a the agent.py
script and a "ghost" state in the AI we are able to avoid problem with the Unity Academy.
