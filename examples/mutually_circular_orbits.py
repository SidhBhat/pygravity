import gravity

# NOTE: Formular for calculating the relative velocity for circular orbits
## v = sqrt((G * (M + m)) / r)
## r = distance between bodies
## M = Mass od one mody
## m = Mass of the other

env = gravity.Environment();

# two bodies in circular orbit around each other
body1 = env.add_body(color='k',mass=5, position=[0,0], velocity=[0,0]);
body2 = env.add_body(color='k',mass=1, position=[2,0], velocity=[0,1.732]);
env.set_radius(0.2);

# plot propertis
env.set_fps(10)
env.set_plot_trace(200);
env.set_bounds(8);
env.set_timestep(0.0001);

# simulate about one period
env.simulate(8);
env.clear_trace();

# simulate one period from the perspective of the more massive body
env.set_origin(body1["position"]);
env.lock_frame_on(body1);
env.simulate(8);
env.unlock_frame();
env.clear_trace();

# change inertial frame
env.set_inertial_frame([0.2,0]);
env.set_bounds(20);
env.simulate(20);
