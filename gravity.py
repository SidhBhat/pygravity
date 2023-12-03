from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegFileWriter
from timeit import default_timer as timer
# from cProfile import label
from types import NoneType
import matplotlib.pyplot as pyplot
import matplotlib
import numpy
import pdb

# Basic class representing a physical body
class Body:
	def _set_vector(vec_like):
		vec = numpy.array(vec_like, dtype=numpy.double);
		numpy.reshape(vec, 2)
		return vec;

	def __init__(self,
			name     = "Body",
			color    = "k",
			mass     = 1,
			radius   = 'Defualt',
			position = [0, 0]
			):

		if(not mass):
			raise ValueError("The mass of a body cannot be zero");
		if (not (type(name) is str and type(color) is str)):
			raise TypeError("\'name\' nad \'color\' expect string arguments only");

		self.position = Body._set_vector(position);
		self.mass     = numpy.double(mass);
		self.name     = name;
		self.color    = color;
		self._Custom_radius = False;

		# special behaviour
		if (type(radius) is str):
			self.radius   = numpy.double(1);
		else:
			self.radius   = numpy.double(radius);
			self._Custom_radius = True;


	def __str__(self):
		return format({
			'name'     : self.name,
			'color'    : self.color,
			'mass'     : self.mass,
			'radius'   : self.radius,
			'position' : self.position
			});

	__repr__ = __str__;

	def __getitem__(self, key):
		if (type(key) is str):
			if (key == 'name'):
				return self.name;
			elif (key == 'color'):
				return self.color;
			elif (key == 'mass'):
				return self.mass;
			elif (key == 'radius'):
				return self.radius;
			elif (key == 'position'):
				return self.position;
			else:
				raise IndexError("Invalid Key value");
		else:
			raise TypeError("Key must be a string value");

	def __setitem__(self, key, value):
		if (type(key) is str):
			if (key == 'name'):
				if (type(value) is str):
					self.name = value;
				else:
					raise TypeError("Body['name'] must be a string value");
			elif (key == 'color'):
				if (type(value) is str):
					self.color = value;
				else:
					raise TypeError("Body['color'] must be a string value");
			elif (key == 'mass'):
				if (value):
					self.mass = numpy.double(value);
				else:
					raise Zero_MassERROR("The mass of a body cannot be zero");
			elif (key == 'radius'):
				if (value):
					self.radius = numpy.double(value);
					self._Custom_radius = True;
				else:
					raise TypeError("The radius of a body cannot be zero");
			elif (key == 'position'):
				self.position = Body._set_vector(value);
			else:
				raise IndexError("Invalid Key value: \'{key}\'");
		else:
			raise TypeError("Key must be a string value");

	def set_parameters(self, **params):
		for key in params.keys():
			self.__setitem__(key, params[key]);


# Body with position, velocity and acceleration
class Dynamic_Body(Body):
	def __init__(self,
			name         = "Body",
			color        = "k",
			mass         = 1,
			radius       = 'Default',
			position     = [0, 0],
			velocity     = [0, 0],
			acceleration = [0, 0]
			):

		Body.__init__(self, name, color, mass, radius, position);
		self.velocity     = Body._set_vector(velocity);
		self.acceleration = Body._set_vector(acceleration);

	_half = numpy.double(0.5);

	def update_velocity(self, tDelta):
		self.velocity += self.acceleration * tDelta;

	def update_position(self, tDelta):
		self.position += \
			(self.velocity + self._half * self.acceleration * tDelta) * tDelta;

	def __str__(self):
		return format({
			'name'         : self.name,
			'color'        : self.color,
			'mass'         : self.mass,
			'radius'       : self.radius,
			'position'     : self.position,
			'velocity'     : self.velocity,
			'acceleration' : self.acceleration
			});

	__repr__ = __str__;

	def __getitem__(self, key):
		if (type(key) is str):
			if (key == 'name'):
				return self.name;
			elif (key == 'color'):
				return self.color;
			elif (key == 'mass'):
				return self.mass;
			elif (key == 'radius'):
				return self.radius;
			elif (key == 'position'):
				return self.position;
			elif (key == 'velocity'):
				return self.velocity;
			elif (key == 'acceleration'):
				return self.acceleration;
			else:
				raise IndexError("Invalid Key value");
		else:
			raise TypeError("Key must be a string value");

	def __setitem__(self, key, value):
		if (type(key) is str):
			if (key == 'name'):
				if (type(value) is str):
					self.name = value;
				else:
					raise TypeError("Body['name'] must be a string value");
			elif (key == 'color'):
				if (type(value) is str):
					self.color = value;
				else:
					raise TypeError("Body['color'] must be a string value");
			elif (key == 'mass'):
				if (value):
					self.mass = numpy.double(value);
				else:
					raise Zero_MassERROR("The mass of a body cannot be zero");
			elif (key == 'radius'):
				if (value):
					self.radius = numpy.double(value);
					self._Custom_radius = True;
				else:
					raise TypeError("The radius of a body cannot be zero");
			elif (key == 'position'):
				self.position     = Body._set_vector(value);
			elif (key == 'velocity'):
				self.velocity     = Body._set_vector(value);
			elif (key == 'acceleration'):
				self.acceleration = Body._set_vector(value);
			else:
				raise IndexError("Invalid Key value");
		else:
			raise TypeError("Key must be a string value");

	def set_parameters(self, **params):
		for key in params.keys():
			self.__setitem__(key, params[key]);


# Iterable containing a list of bodies
class Body_list:
	"""argument to __init__ 'Body_list' is a list of dictionaries of the form:
	{
	'name':     "<name>",   # a string
	'color':    "<color>",  # single character
	'radius':    <radius>,
	'position': [<x_pos>, <y_pos>],
	'velocity': [<x_vel>, <y_vel>]
	}
	"""
	def __init__(self, *body_list):  # Expects a list of dictionaries
		self.body_list = [];

		n = int(1);
		for body in body_list:
			self.body_list.append(Dynamic_Body(
				name         = body.get('name', f"Body {n}"),
				color        = body.get('color', 'k'),
				mass         = body.get('mass', 1),
				radius       = body.get('radius', 'Defualt'),
				position     = body.get('position', [0, 0]),
				velocity     = body.get('velocity', [0, 0])
				));
			n += 1;

	# with syntax add_body(mass=1, position=[1,0], ...)
	def add_body(self, **body):
		self.body_list.append(Dynamic_Body(
				name         = body.get('name', f"Body {len(self.body_list) + 1}"),
				color        = body.get('color', 'k'),
				mass         = body.get('mass', 1),
				radius       = body.get('radius', 'Default'),
				position     = body.get('position', [0, 0]),
				velocity     = body.get('velocity', [0, 0])
				));
		return self.body_list[-1];

	# specify index or name of body
	def remove_body(self, body_id):
		if (type(body_id) is str):
			index_list = [];
			n = int(0);
			for body in self:
				if (body.name == body_id):
					index_list.append(n);
				n += int(1);
			if (not len(index_list)):
				raise ValueError(f"Body with name \'{body_id}\' not found");
			index_list.sort(reverse=True);
			for index in index_list:
				self.body_list.pop(index);
		elif (type(body_id) is int):
			self.body_list.pop(body_id);
		else:
			raise IndexError(f"Specifed body_id \'{body_id}\' not found");

	def get_body(self, body_id, get_index=False):
		get_index  = bool(get_index);
		index_list = [];
		if (type(body_id) is str):
			n = int(0);
			for body in self:
				if (body_id == body.name):
					index_list.append(n);
				n += int(1);
			if (len(index_list)):
				if (len(index_list) == 1):
					if (get_index):
						return index_list[0];
					else:
						return self[index_list[0]];
				else:
					if (get_index):
						return index_list;
					else:
						body_list = [];
						for index in index_list:
							body_list.append(self[index]);
						ret_val = Body_list()
						ret_val.body_list = body_list;
						return ret_val;
			else:
				raise ValueError(f"Body with name \'{body_id}\' not found");
		elif (type(body_id) is int):
			if (body_id >= len(self)):
				raise IndexError("Index out of range");
			return self.body_list[body_id];
		else:
			raise TypeError(f"Invalid key \'{body_id}\'");

	def __len__(self):
		return len(self.body_list);

	def __getitem__(self, key):
		return self.get_body(key);

	# Required as we frequently use the iterable in nested loops
	class _Iterator:
		def __init__(self, body_list_obj):
			self._data = body_list_obj;
			self._pos = int(0);
		def __next__(self):
			if (self._pos >= len(self._data)):
				raise StopIteration;
			val = self._data[self._pos];
			self._pos += int(1);
			return val;

	def __iter__(self):
		return self._Iterator(self);


# Plot and simulation environment
class Environment(Body_list):
	"""argument to __init__ 'Environment' is a list of dictionaries of the form:
	{
	'name':     "<name>",   # a string
	'color':    "<color>",  # single character
	'radius':    <radius>,
	'position': [<x_pos>, <y_pos>],
	'velocity': [<x_vel>, <y_vel>]
	}
	"""
	def __init__(self,
			*body_list, # Expects a list of dictionaries
			**plot_properties):

		# initialise Body_list
		if (len(body_list)):
			Body_list.__init__(self, *body_list);
		else:
			Body_list.__init__(self);

		# initialise data members
		self.figure, self.axes = (None, None);
		self.plot_properties = self._Plot_properties();
		self.set_simulation_opts(**plot_properties);

	class _Plot_properties:
		#default values
		def __init__(self):
			# the timestep interval
			self.timestep       = numpy.double(0.0001);
			self.fps            = numpy.double(20);
			self.radius         = numpy.double(0.1);
			self.bounds         = numpy.double(5);
			self.origin         = numpy.array([0,0], dtype=numpy.double);
			self.trace_length   = numpy.double(100);
			self.G              = numpy.double(1);
			# Lock onto a body
			self.frame_lockon   = None;
			# time wrap
			self.timewarp       = numpy.double(1);
			# intertial fram velocity (w.r.t zero momentum frame)
			self.frame_velocity = numpy.array([0,0], dtype=numpy.double);
			# Use RK4
			self.use_rk4        = False;
			# bring vector to correct shape
			numpy.reshape(self.origin, 2);
			numpy.reshape(self.frame_velocity, 2);

	class Time:
		minute = numpy.double(60);
		hour   = numpy.double(3600);
		day    = numpy.double(86400);
		month  = numpy.double(2592000);
		year   = numpy.double(31536000);

		def __init__(self, time, time_unit='seconds'):
			if (type(time_unit) is str):
				if(time_unit == 'seconds' or time_unit == 'second' or
					time_unit == 'sec' or time_unit == 's'):

					self.time = numpy.double(time);
				elif(time_unit == 'minutes' or time_unit == 'minute' or
					time_unit == 'min'):

					self.time = self.minute * numpy.double(time);
				elif(time_unit == 'hours' or time_unit == 'hour' or
					time_unit == 'hr' or time_unit == 'h'):

					self.time = self.hour * numpy.double(time);
				elif(time_unit == 'days' or time_unit == 'day' or
					time_unit == 'd'):

					self.time = self.day * numpy.double(time);
				elif(time_unit == 'months' or time_unit == 'month' or
					time_unit == 'mo'):

					self.time = self.month * numpy.double(time);
				elif(time_unit == 'years' or time_unit == 'year' or
					time_unit == 'yr' or time_unit == 'y'):

					self.time = self.year * numpy.double(time);
				else:
					raise ValueError(f"Unrecognised time unit: \'{time_unit}\'");
			else:
				raise ValueError("\'time_unit\' must be a string type");

		def get_time(self):
			return self.time;

		def time_string(time):
			one    = numpy.double(1.1);
			minute = Environment.Time.minute;
			hour   = Environment.Time.hour;
			day    = Environment.Time.day;
			month  = Environment.Time.month;
			year   = Environment.Time.year;
			suffix = "";

			if(time / year > one):
				time = time / year;
				suffix    = "years";
			elif(time / month > one):
				time = time / month;
				suffix    = "months";
			elif(time / day > one):
				time = time / day;
				suffix    = "days";
			elif(time / hour > one):
				time = time / hour;
				suffix    = "hours";
			elif(time / minute > one):
				time = time / minute;
				suffix    = "minutes";
			else:
				suffix    = 's';

			return ("time = %0.1f " + suffix) % time;

		def get_time_string(self):
			return self.time_string(self.time);

	# call Signature:
	## set_simulation_opts(timestep=<timestep>, fps=<fps>, radius=<radius>,
	##       bounds=<bounds>, origin=<[x0, y0]>, trace=<trace>, lock=<body>,
	##       rk4=<bool>, frame=<inertial_frame>)
	def set_simulation_opts(self, **opts):
		for key in opts.keys():
			if(type(key) is str):
				if(key == 'timestep' or key == 'tdelta' or key == 'ts'):
					self.set_timestep(opts[key]);
				elif(key == 'fps'):
					self.set_fps(opts[key]);
				elif(key == 'radius'):
					self.set_radius(opts[key]);
				elif(key == 'bound' or key == 'lim' or key == 'limit' or key == 'axis_limit'):
					self.set_bounds(opts[key]);
				elif(key == 'origin' or key == 'center' or key == 'centre'):
					self.set_origin(opts[key]);
				elif(key == 'trace' or key == 'tr' or key == 'trace_length'):
					self.set_plot_trace(opts[key]);
				elif(key == 'G'):
					self.set_gravitational_constant(opts[key]);
				elif(key == 'lock' or key == 'lockon'):
					self.lock_frame_on(opts[key]);
				elif(key == 'rk4' or key == 'use_rk4' or key == 'userk4'):
					self.lock_frame_on(opts[key]);
				elif(key == 'frame' or key == 'inertial_frame'):
					self.set_inertial_frame(opts[key]);
				elif(key == 'timewarp'):
					self.set_timewarp(opts[key]);
				else:
					raise IndexError(f"Invalid Key Value \'{key}\'");
			else:
				raise TypeError("key must be of type string");

	def set_timestep(self, timestep):
		self.plot_properties.timestep = numpy.double(timestep);

	def set_fps(self, fps):
		self.plot_properties.fps = numpy.double(fps);

	def set_inertial_frame(self, velocity):
		self.plot_properties.frame_velocity = \
			numpy.array(velocity, dtype=numpy.double);
		numpy.reshape(self.plot_properties.frame_velocity, 2);

	def set_radius(self, radius):
		self.plot_properties.radius = numpy.double(radius);

	def set_bounds(self, lim):
		self.plot_properties.bounds = numpy.double(lim);

	def set_origin(self, origin):
		self.plot_properties.origin = numpy.array(origin, dtype=numpy.double);
		numpy.reshape(self.plot_properties.origin, 2);

	def set_plot_trace(self, trace):
		self.plot_properties.trace_length = numpy.double(trace);

	def set_gravitational_constant(self, Num):
		self.plot_properties.G = numpy.double(Num);

	def set_timewarp(self, timewarp):
		timewarp = numpy.double(timewarp);
		if (timewarp < numpy.double(0)):
			raise ValueError("You cannot reverse time like this :v");
		self.plot_properties.timewarp = numpy.double(timewarp);

	def lock_frame_on(self, planet):
		if(type(planet) is str):
			planet = self.get_body(planet, get_index=False);
			if (type(planet) is Body_list):
				raise ValueError("More than one {planet}");

		if(not (Body is type(planet) or Dynamic_Body is type(planet))):
			raise TypeError(f"\'{planet}\' is not a body type")

		for body in self:
			if (body is planet):
				self.plot_properties.frame_lockon = planet;
				return;
		raise ValueError("Body is not contained in Environment");

	def use_rk4(self, opt=True):
		opt = bool(opt);
		if(opt):
			self.plot_properties.use_rk4 = True;
		else:
			self.plot_properties.use_rk4 = False;

	def get_timestep(self):
		return self.plot_properties.timestep;

	def get_fps(self):
		return self.plot_properties.fps;

	def get_inertial_frame(self):
		return self.plot_properties.frame_velocity;

	def get_radius(self):
		return self.plot_properties.radius;

	def get_bounds(self):
		return self.plot_properties.bounds;

	def get_origin(self):
		return self.plot_properties.origin;

	def get_gravitational_constant(self):
		return self.plot_properties.G;

	def get_timewarp(self):
		return self.plot_properties.timewarp;

	def unlock_frame(self):
		self.plot_properties.frame_lockon = None;

	def clear_trace(self):
		for body in self:
			if (hasattr(body, 'trace')):
				del body.trace, body.xtrace, body.ytrace;

	def show(self, **properties):
		# initialise figure
		self.figure, self.axes = pyplot.subplots(1,1);
		self.set_simulation_opts(**properties);
		# set plot window limits
		xllim = self.plot_properties.origin[0] - self.plot_properties.bounds / 2;
		xhlim = self.plot_properties.origin[0] + self.plot_properties.bounds / 2;
		yllim = self.plot_properties.origin[1] - self.plot_properties.bounds / 2;
		yhlim = self.plot_properties.origin[1] + self.plot_properties.bounds / 2;

		# initialise _Compute instance
		compute = _Compute(self);
		# plot
		compute.setup_bodies();
		compute.plot_setup();
		compute.plot_bodies();

		# set axis properties
		self.axes.axis('square');
		self.axes.set_xlim(xllim, xhlim);
		self.axes.set_ylim(yllim, yhlim);
		self.axes.autoscale(False);
		self.axes.axis('off');

		pyplot.show();

	# alias functions
	draw = show;

	def simulate(self, time, time_unit='seconds', save=None, dpi=100, extra_args=None,
			plot_simulation_result=False, **properties):

		class Simulation_Data:
			def __init__(self):
				self.time_parameter   = [];
				self.momentum         = [];
				self.angular_momentum = [];
				self.energy           = [];
				self.potential_energy = [];
				self.kinetic_energy   = [];

			def plot(self):
				# initialise figure
				self.figure, self.axes = pyplot.subplots(1,1);
				# plot data
				self.axes.plot(self.time_parameter,
				   self.momentum, label="total momentum");
				self.axes.plot(self.time_parameter,
				   self.angular_momentum, label="total angular_momentum");
				self.axes.plot(self.time_parameter,
				   self.potential_energy, label="potential_energy");
				self.axes.plot(self.time_parameter,
				   self.kinetic_energy, label="kinetic_energy");
				self.axes.plot(self.time_parameter,
				   self.energy, label="total energy");
				# set axis properties
				self.axes.grid();
				self.axes.legend()

				pyplot.show();

		# initialise figure
		self.set_simulation_opts(**properties);
		self.figure, self.axes = pyplot.subplots(1,1);
		# initialise _Compute instance
		compute = _Compute(self);

		# set plot window limits
		xllim = self.plot_properties.origin[0] - self.plot_properties.bounds / 2;
		xhlim = self.plot_properties.origin[0] + self.plot_properties.bounds / 2;
		yllim = self.plot_properties.origin[1] - self.plot_properties.bounds / 2;
		yhlim = self.plot_properties.origin[1] + self.plot_properties.bounds / 2;

		# initialise Variables
		plot_simulation_result = bool(plot_simulation_result);
		write_to_file = True;
		timestep      = self.get_timestep();
		timewarp      = self.get_timewarp();
		fps           = self.get_fps() / timewarp;
		time_text     = None;
		time_data     = [];

		# initialise local class instances
		if (plot_simulation_result):
			# initialise info panel
			info_panel = Simulation_Data();
		# initialise time class
		time = Environment.Time(time, time_unit).get_time();
		# function to convert time into a string with human readable units
		time_string = Environment.Time.time_string;

		if(fps * timestep >= numpy.double(1)):
			raise ValueError(f"timestep \'{timestep}\' is too large");

		if(fps * time <= numpy.double(1)):
			raise ValueError(f"simulation time is lesser than the interval between frames"
							+ f"\nMinimum reqired time: {1 / fps}");

		# Writer = FFMpegFileWriter(fps=10);
		# Writer.setup(self.figure, "animation.mp4");

		def setup():
			nonlocal time_string, time_text, compute;
			# setup plot
			compute.setup_bodies();
			compute.plot_setup();
			# add time text
			time_text = self.axes.text(0.05, 0.9, time_string(numpy.double(0.0)),
							transform = self.axes.transAxes);
			compute.add_artist(time_text);
			artists = compute.plot_bodies();
			return artists;

		def animate(frame_data):
			nonlocal time_string, compute, time_text;
			start = timer();
			# plot bodies
			if(self.plot_properties.use_rk4):
				compute.update_bodies_RK4(1 / fps);
			else:
				compute.update_bodies(1 / fps);
			# update time text
			time_text.set_text(time_string(frame_data / fps));
			# update artists
			artists = compute.plot_bodies();
			# collect simulation data
			if(plot_simulation_result):
				info_panel.time_parameter.append(frame_data / fps);
				info_panel.momentum.append(compute.total_momentum());
				info_panel.angular_momentum.append(compute.total_angular_momentum());
				info_panel.kinetic_energy.append(compute.total_kinetic_energy());
				info_panel.potential_energy.append(compute.total_potential_energy());
				info_panel.energy.append(compute.total_energy());
			# collect simulation time
			# self.figure.canvas.flush_events()
			# Writer.grab_frame();
			print(time_string(frame_data / fps), end='\r');
			time_data.append(timer() - start);
			return artists;

		# set axis properties
		self.axes.axis('square');
		self.axes.set_xlim(xllim, xhlim);
		self.axes.set_ylim(yllim, yhlim);
		self.axes.autoscale(False);
		self.axes.axis('off');

		ani = FuncAnimation(self.figure, animate,
			numpy.arange(1, fps*time, dtype=numpy.double),
			init_func=setup, interval=1000 / fps / timewarp, repeat=False, blit=False);

		# FFwriter = FFMpegFileWriter(fps=self.get_fps(), extra_args=['-c:v', 'h264_nvenc','-preset', 'slow', '-vf', f'fps={self.get_fps()}'])
		# ani.save('basic_animation.mp4', writer=FFwriter, dpi=500);

		pyplot.show();
		# Writer.finish();

		if(plot_simulation_result):
			info_panel.plot();

		loop_interval        = sum(time_data) / len(time_data);
		theoretical_interval = 1.0 / fps / timewarp;
		actual_interval      = theoretical_interval + loop_interval;
		print(f"Mean time taken to simulate one frame:\t{loop_interval}");
		print(f"Time between frames as per fps setting:\t{theoretical_interval}");
		print(f"Real Time Ratio :\t{theoretical_interval / actual_interval}");
		# print(f"Total Simulation Time: {simulation_time}");


# Helper class for computations
class _Compute(Environment):
	# perform a shallow copy of the Environment instance
	def __init__(self, Env):
		#initialise the Body_list members.
		self.body_list         = Env.body_list;

		# initialise Environment members
		self.figure, self.axes = Env.figure, Env.axes;
		self.plot_properties   = Env.plot_properties;

		# Fix min mass
		self._min_mass         = min(body.mass for body in self.body_list);

		# redundant
		assert self.body_list is Env.body_list;
		assert self.figure    is Env.figure;
		assert self.axes      is Env.axes
		assert self.plot_properties  is Env.plot_properties;

	## Method for global manipulation

	# generator to get a list of all other bodies
	def every_other_body(self, body):
		n = int(0);

		while(n < len(self)):
			bd = self[n];
			if (not body is bd):
				yield bd;
			n += 1;

	# return the least massive body's mass
	def min_mass(self):
		return self._min_mass;

	# return velocity such that there is no net momentum
	def zero_momentum_frame(self):
		Mass = numpy.double(0);
		velocity = numpy.zeros(2);

		for body in self:
			Mass += body.mass;
			velocity += body.mass * body.velocity;
		return (1/Mass) * velocity;

	# populate acceleration data with respect to frame of referance
	def populate_acceleration_data(self):
		for body in self:
			self.update_acceleration(body);

	# adjust the radius relative to minimum mass
	def adjust_radii(self):
		for body in self:
			if (not body._Custom_radius):
				body.radius = (body.mass / self._min_mass) ** 0.5 * \
					self.plot_properties.radius;

	# change the inertial frame by velocity (w.r.t current frame)
	def change_intertial_frame(self, velocity):
		velocity = numpy.array(velocity, dtype=numpy.double);
		numpy.reshape(velocity,2);

		for body in self:
			body.velocity -= velocity;

	def total_potential_energy(self):
		return sum(self.get_potential_energy(body) for body in self);

	def total_kinetic_energy(self):
		norm   = numpy.linalg.norm;
		return sum(body.mass * norm(body.velocity) ** 2 / 2 for body in self);

	def total_energy(self):
		return self.total_potential_energy() + self.total_kinetic_energy();

	def total_angular_momentum(self):
		return sum(body.mass * _Compute._cross(body.position, body.velocity) for body in self);

	def total_momentum(self):
		norm   = numpy.linalg.norm;
		return norm(sum(body.mass * body.velocity for body in self));

	## methods for manipulating and getting information on indivisual bodies

	# get the radius of a body
	def radius(self, body):
		return body.radius;

	# return true if bd1 and bd2 overlap
	def two_bodies_overlap(self, bd1, bd2):
		norm = numpy.linalg.norm;

		if (bd1 is bd2):
			raise ValueError("Cannot calculate overlap for the same body");

		if (norm(bd1.position - bd2.position) <= self.radius(bd1) + self.radius(bd2)):
			return True;
		return False;

	# merge bodies bd1 and bd2 d in a perfectly inelastic collision (momentum conseving)
	def merge_two_bodies(self, bd1, bd2):
		if (bd1 is bd2):
			raise ValueError("Cannot merge the same body onto itself");
		locked_body = self.plot_properties.frame_lockon;
		delete_string = '__TO_BE_DELETED__';
		Mass     = bd1.mass + bd2.mass;
		velocity = (1 / Mass) * (bd1.mass * bd1.velocity + \
					bd2.mass * bd2.velocity);

		if (bd2.mass >= bd1.mass):
			if (bd1 is locked_body):
				self.plot_properties.frame_lockon = bd2;
			bd2.mass = Mass;
			bd2.velocity = velocity;
			bd1.name = delete_string;
			# remove the artist object associated with teh body
			if(hasattr(bd1,'circle')):
				bd1.circle.remove();
			bd = bd2;
		else:
			if (bd2 is locked_body):
				self.plot_properties.frame_lockon = bd1;
			bd1.mass = Mass;
			bd1.velocity = velocity;
			bd2.name = delete_string;
			# remove the artist object associated with teh body
			if(hasattr(bd2,'circle')):
				bd2.circle.remove();
			bd = bd1;
		self.remove_body(delete_string);

		return bd;

	# return the potential energy of a body
	def get_potential_energy(self, body):
		G = self.plot_properties.G;
		norm   = numpy.linalg.norm;
		power  = numpy.power;

		return -body.mass * sum(G * bd.mass / norm(bd.position - body.position)
						  for bd in self.every_other_body(body));

	# get acceleration of body irrespective of referance frame (non-inertial)
	def get_acceleration(self, body):
		G = self.plot_properties.G;
		norm   = numpy.linalg.norm;
		power  = numpy.power;

		return sum((G * bd.mass / power(norm(bd.position - body.position),3)) * \
						(bd.position - body.position)
						for bd in self.every_other_body(body));

	# update the acceleration on agiven body
	## respects non inertial frames which are locked to a specific body
	def update_acceleration(self, body):
		locked_body = self.plot_properties.frame_lockon;
		G           = self.plot_properties.G;
		norm        = numpy.linalg.norm;
		power       = numpy.power;

		if (body is locked_body):
				body.acceleration = numpy.array([0,0], dtype=numpy.double);
				return;

		body.acceleration = \
					sum((G * bd.mass / power(norm(bd.position - body.position),3)) * \
					(bd.position - body.position)
					for bd in self.every_other_body(body));
		if (not (type(locked_body) is NoneType)):
			body.acceleration -= self.get_acceleration(locked_body);

	## Vector manipulation
	def _dot(vec1, vec2):
		return vec1[0] * vec2[0] + vec1[1] * vec2[1];

	def _cross(vec1, vec2):
		return vec1[0] * vec2[1] - vec1[1] * vec2[0];

	def _set_vec_length(vec, length):
		norm   = numpy.linalg.norm;
		if (vec.any()):
			return (length / norm(vec)) * vec;
		else:
			raise ValueError("Vector is zero");

	## Simulation Methods

	# the referance frame is set to center of mass if frame lock is unspecified
	# else it is locked to a specific body (non inertial)
	def _refresh_referance_frame(self):
		if (type(self.plot_properties.frame_lockon) is NoneType):
			self.change_intertial_frame(self.zero_momentum_frame()
							   + self.plot_properties.frame_velocity);
		else:
			self.change_intertial_frame(
				self.plot_properties.frame_lockon.velocity);
			self.plot_properties.frame_lockon.acceleration = \
				numpy.array([0,0], dtype=numpy.double);
			self.plot_properties.frame_lockon.velocity     = \
				numpy.array([0,0], dtype=numpy.double);

	# routine for handling collisions
	def _collision_handler(self, body):
		for bd in self.every_other_body(body):
			if(self.two_bodies_overlap(body, bd)):
				self.merge_two_bodies(body,bd);
				self.adjust_radii();
				self._refresh_referance_frame();

	# update body
	def _update_body(self, body, tDelta):
		self.update_acceleration(body);
		body.update_position(tDelta);
		body.update_velocity(tDelta);

	# update body using RK4 (Classical Runge Kutta)
	def _update_body_RK4(self, body, tDelta):
		initial_position = body.position;
		initial_velocity = body.velocity;

		def get_acceleration(position):
			nonlocal self;
			locked_body = self.plot_properties.frame_lockon;
			G           = self.plot_properties.G;
			norm        = numpy.linalg.norm;
			power       = numpy.power;
			acceleration = \
				sum((G * bd.mass / power(norm(bd.position - position),3)) * \
				(bd.position - position)
				for bd in self.every_other_body(body));
			if (not (type(locked_body) is NoneType)):
				acceleration -= self.get_acceleration(locked_body);
			return acceleration;

		def get_velocity(tDelta, acceleration):
			nonlocal initial_velocity;
			return initial_velocity + tDelta * acceleration;

		def get_position(tdelta, velocity, acceleration):
			nonlocal initial_position;
			return initial_position + tDelta * (velocity + acceleration * tDelta / 2);

		k1 = get_acceleration(initial_position);
		l1 = initial_velocity;

		k2 = get_acceleration(get_position(tDelta / 2, l1, k1));
		l2 = get_velocity(tDelta / 2, k1);

		k3 = get_acceleration(get_position(tDelta / 2, l2, k2));
		l3 = get_velocity(tDelta / 2, k2);

		k4 = get_acceleration(get_position(tDelta, l3, k3));
		l4 = get_velocity(tDelta, k3);

		body.position = initial_position + tDelta/6 * (l1 + 2*l2 + 2*l3 + l4);
		body.velocity = initial_velocity + tDelta/6 * (k1 + 2*k2 + 2*k3 + k4);

	# this routine will setup the bodies up for simulation, Specifically:
	## overlapping bodies are merged...
	## the intertial frame is set to the net zero momentum frame (if applicable)
	def setup_bodies(self):
		self.adjust_radii();
		for body in self:
			for bd in self.every_other_body(body):
				if(self.two_bodies_overlap(body,bd)):
					self.merge_two_bodies(body,bd);
		self.populate_acceleration_data();
		self._refresh_referance_frame();

	# update body to it's position t=time in the future
	def update_bodies(self, time):
		tDelta = self.get_timestep();
		time   = numpy.abs(numpy.double(time));

		t = numpy.double(0);
		while (t < time):
			for body in self:
				self._collision_handler(body);
			for body in self:
				self._update_body(body, tDelta);
			t += tDelta;

	# update body to it's position t=time in the future
	def update_bodies_RK4(self, time):
		tDelta = self.get_timestep();
		time   = numpy.double(time);

		t = numpy.double(0);
		while (t < time):
			for body in self:
				self._collision_handler(body);
			for body in self:
				self._update_body_RK4(body, tDelta);
			t += tDelta;

	## plot methods

	# setup plot
	def plot_setup(self):
		if(not hasattr(self, '_Circles_Collection')):
			self._Circles_Collection = None;
		for body in self:
			# prevent previous trace being erased
			if (not hasattr(body,'trace')):
				body.xtrace = [];
				body.ytrace = [];
			body.trace, = self.axes.plot([],[], body.color+'.', lw=1, ms=1);
			body.circle = \
				pyplot.Circle(body.position, radius=self.radius(body),
					color=body.color);
		## Generate list of artists
		# Bodies = self.axes.add_collection(
		# 	matplotlib.collections.PatchCollection(tuple(
		# 		body.circle for body in self), match_original=True));
		Bodies = list(self.axes.add_patch(body.circle) for body in self);
		Artists = list(body.trace for body in self) + Bodies;
		# Artists.append(Bodies);
		self.artists = Artists;

	# update the frame
	def plot_bodies(self):
		for body in self:
			if (len(body.xtrace) < self.plot_properties.trace_length):
				body.xtrace.append(body.position[0]);
				body.ytrace.append(body.position[1]);
			else:
				body.xtrace.pop(0);
				body.ytrace.pop(0);
				body.xtrace.append(body.position[0]);
				body.ytrace.append(body.position[1]);
			body.trace.set_data(body.xtrace, body.ytrace);
			body.circle.set_center(body.position);
			body.circle.set_radius(body.radius);

		return self.artists;

	# add an external artist to the list of artists
	# You will have to manage updating them yourself
	def add_artist(self, artist):
		if (not issubclass(type(artist),  matplotlib.artist.Artist)):
			raise TypeError("artist must be a instace of \'matplotlib.artist.Artist\'");
		if (not hasattr(self, 'artists')):
			raise RuntimeError("Please execute \'plot_setup\' before adding Artists");
		self.artists.append(artist);
		return self.artists;

