#### Sample Code ###

import gravity
# import pdb # for debugging

env = gravity.Environment();

## 10 bodies placed on a diagonal
# for i in range (-5,5,1):
# 	env.add_body(mass=1,position=[i,i]);

# Three bodies orbiting each other
env.add_body(name='earth',color='b',mass=10,position=[0,0],velocity=[0.181,0]);
env.add_body(name='moon',color='k',mass=1,position=[0,3],velocity=[-1.769,0]);
env.add_body(name='satelite',color='r',mass=0.1,position=[0,3.5],velocity=[-0.355,0]);

## Collision test
# env.add_body(name='body 1',color='b',mass=100,position=[0,0],velocity=[0,0]);
# env.add_body(name='body 2',color='r',mass=100,position=[-2,0],velocity=[0,7]);
# env.add_body(name='body 3',color='g',mass=50,position=[4,0],velocity=[-7,0.1]);

## NOTE: at extreme velocities and high gravitational forces the inaccuracies of
## euler integration can be observed...and orbital energy is slowly
## depleted.
### METHODS TO REDUCE inaccuracies:
###   - decrease value of G
###   - scale 'up' the system
###   - decrease the mass involed
### and eny other technique that would reduce the rate of change in accelration
### and the magnitude of G force
# env.add_body(name='black hole',color='k',mass=100000,position=[0,0],velocity=[0,0]);
# env.add_body(name='satelite',color='r',mass=10,position=[10,0],velocity=[0,80]);
# env.add_body(name='satelite',color='r',mass=1,position=[-10,0],velocity=[0,95]);

env.set_fps(60);
env.set_markersize(30);
env.set_timestep(0.0001);
env.set_bounds(5);

env.simulate(100);
