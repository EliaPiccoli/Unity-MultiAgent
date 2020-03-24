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

To run there are two options:
  1. Using the Unity Editor: is the default set up, run in a terminal python3 main.py the press Play in the editor in
     which the project Maze_Multi_3x3 is open.
  2. Create the GameApp: create the game app of the project Maze_Multi_3x3. In main.py set up env_name to the path
     to get to the game from the main folder and put it instead of None in the following line, then run python3 main.py and it
     will start the training.
