import gravity

env = gravity.Environment();

## NOTE Majority tests were done with this example as it tends to show
## remarkable unstability  at lower timsteps.

## Bodies that settle in an aliptical orbit after collision of "satelite"
## with body 1
env.add_body(name='body 1',color='b',mass=200,position=[2,0],velocity=[0,0]);
env.add_body(name='body 2',color='r',mass=200,position=[-2,0],velocity=[0,6]);
satelite = env.add_body(name='body 3',color='g',mass=50,position=[4,0],velocity=[-7,0.1]);


# Set plot properties
env.set_fps(60);
env.set_radius(0.05);
env.set_timestep(0.00001);
env.set_bounds(8);
env.set_origin([0,0]);
# env.use_rk4();

# simulate
env.simulate(10, plot_simulation_result=True);
