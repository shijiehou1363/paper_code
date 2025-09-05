
## RLProcess
gsar_tree.py calls tree class based on compiled c file to interact with RL agent that using AC and PPO

```python
python RLProcess/gsar_tree.py --welltrained True    # for testing well trained agent model
python RLProcess/gsar_tree.py --mode 1              # for training
python RLProcess/gsar_tree.py                       # for testing 
```
## model
trained RL agent model, the example is using synthetic normal data
## rtree
base tree implementation, including basic class structure, operations of builing a tree, retrieve state info of nodes' subtree...
