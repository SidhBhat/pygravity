import gravity
import random
import math

env = gravity.Environment();

# Set Environment
env.set_gravitational_constant(6.6743e-11);
max_mass      = 1.0e25;
min_mass      = 1.0e20;
system_radius = 1.0e10;
number        = 30;
Mass          = 0
max_velocity = 300
for i in range(number):
	r = random.uniform(0, system_radius)
	t = random.uniform(0, 2 * math.pi);
	x = r * math.cos(t);
	y = r * math.sin(t);
	r = random.uniform(0, max_velocity);
	vx = r * math.sin(t);
	vy = - r * math.cos(t);
	tmp = env.add_body(
		mass     = random.uniform(1, max_mass),
		position = [x,y],
		velocity = [vx, vy]
		);
	Mass += tmp.mass;

# for body in env:
# 	r = random.uniform(0, max_velocity);
# 	t = random.uniform(0, 2 * math.pi);
# 	vx = r * math.sin(t);
# 	vy = - r * math.cos(t);
# 	body['velocity'] = [vx, vy];
env.set_radius(2.0e5)

env.set_bounds(2 * system_radius);
env.set_origin([0,0]);

env.draw();

# for bd in env:
# 	# print(math.sqrt(bd.velocity[0]**2 + bd.velocity[0]**2))
# 	print(bd);

env.set_timestep(1000);
env.set_fps(20);
env.set_timewarp(100000)
env.simulate(100,'days');

for bd in env:
	# print(math.sqrt(bd.velocity[0]**2 + bd.velocity[0]**2))
	print(bd);

env.draw();
