from types import NoneType
import matplotlib
import matplotlib.pyplot as pyplot
import numpy
from matplotlib.animation import FuncAnimation
import pdb

# Execption Classes
class Zero_MassERROR(Exception):
	pass;

# Not meant to be used directly
class Body:
	def _set_vector(vec_like):
		vec = numpy.array(vec_like, dtype=numpy.double);
		numpy.reshape(vec,2)
		return vec;

	def __init__(self,
			name     = "Body",
			color    = "k",
			mass     = 1,
			radius   = 'Defualt',
			position = [0, 0]
			):

		if(not mass):
			raise Zero_MassERROR("The mass of a body cannot be zero");
		if (not (type(name) is str and type(color) is str)):
			raise ValueError("\'name\' nad \'color\' expect string arguments only");

		self.position = Body._set_vector(position);
		self.mass     = numpy.double(mass);
		self.name     = name;
		self.color    = color;
		self._Costom_radius = False;

		# special behaviour
		if (type(radius) is str):
			self.radius   = numpy.double(1);
		else:
			self.radius   = numpy.double(radius);
			self._Costom_radius = True;


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
					self._Costom_radius = True;
				else:
					raise TypeError("The radius of a body cannot be zero");
			elif (key == 'position'):
				self.position = Body._set_vector(value);
			else:
				raise IndexError("Invalid Key value: \'{key}\'");
		else:
			raise TypeError("Key must be a string value");

# Not meant to be used directly
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
					self._Costom_radius = True;
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

# Iterable containing all bodies
class Body_list:
	"""argument to __init__ 'body_list' is a list of dictionaries of the form:
	{
	'name':     "<name>",   # a string
	'color':    '<color>',  # single character
	'mass':     <mass>,
	'position': [<x_pos>, <y_pos>],
	'velocity': [<x_vel>, <y_vel>]
	}
	"""
	def __init__(self,
			*body_list # Expects a list of dictionaries
			):
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
			for body in self.body_list:
				if (body.name == body_id):
					index_list.append(n);
				n += 1;
			if (not len(index_list)):
				raise ValueError(f"Body with name \'{body_id}\' not found");
			index_list.sort(reverse=True);
			for index in index_list:
				self.body_list.pop(index);
		elif (type(body_id) is int):
			self.body_list.pop(body_id);
		else:
			raise IndexError(f"Specifed body_id \'{body_id}\' not found");

	def __len__(self):
		return len(self.body_list);

	def __getitem__(self, index):
		if (not type(index) is int):
			raise TypeError("Indices must be integers only");
		elif (index >= len(self)):
			raise IndexError("Index out of range");
		return self.body_list[index];

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
	"""argument to __init__ 'body_list' is a list of dictionaries of the form:
	{
	'name':     "<name>",   # a string (optional)
	'color':    '<color>',  # single character (optional)
	'mass':     <mass>,
	'position': [x_pos, y_pos],
	'velocity': [x_vel, y_vel],
	}
	"""
	def __init__(self,
			*body_list # Expects a list of dictionaries
			):
		if (len(body_list)):
			Body_list.__init__(self, *body_list);
		else:
			Body_list.__init__(self);

		# initialise data members
		self.figure, self.axes = (None, None);
		self.plot_properties = self._Plot_properties();

	class _Plot_properties:
		#default values
		def __init__(self):
			# the timestep interval
			self.timestep     = numpy.double(0.0001);
			self.fps          = numpy.double(20);
			self.radius       = numpy.double(0.1);
			self.bounds       = numpy.double(5);
			self.origin       = numpy.array([0,0], dtype=numpy.double);
			self.trace_length = numpy.double(100);
			self.G            = numpy.double(1);
			# bring vector to correct shape
			numpy.reshape(self.origin,2);

			# Lock onto a body
			self.frame_lockon = None;

	def draw(self, **properties):
		self.figure, self.axes = pyplot.subplots(1,1);
		self.set_simulation_opts(**properties);
		xllim = self.plot_properties.origin[0] - self.plot_properties.bounds / 2;
		xhlim = self.plot_properties.origin[0] + self.plot_properties.bounds / 2;
		yllim = self.plot_properties.origin[1] - self.plot_properties.bounds / 2;
		yhlim = self.plot_properties.origin[1] + self.plot_properties.bounds / 2;

		compute = _Compute(self);
		# compute.populate_acceleration_data();
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
	show = draw;

	def simulate(self, time, **properties):
		self.set_simulation_opts(**properties);
		self.figure, self.axes = pyplot.subplots(1,1);
		compute                = _Compute(self);
		xllim = self.plot_properties.origin[0] - self.plot_properties.bounds / 2;
		xhlim = self.plot_properties.origin[0] + self.plot_properties.bounds / 2;
		yllim = self.plot_properties.origin[1] - self.plot_properties.bounds / 2;
		yhlim = self.plot_properties.origin[1] + self.plot_properties.bounds / 2;
		time_template = 'time = %.1fs';
		timestep      = self.get_timestep();
		fps           = self.get_fps();
		time_text     = None;

		def setup():
			nonlocal time_text, time_template, compute;
			compute.setup_bodies();
			compute.plot_setup();
			time_text = self.axes.text(0.05, 0.9, (time_template % 0.0),
							  transform = self.axes.transAxes);
			artists = compute.plot_bodies();
			artists.append(time_text);
			return artists;

		def animate(frame_data):
			compute.update_bodies(1 / fps);
			time_text.set_text(time_template % (frame_data / fps));
			artists = compute.plot_bodies();
			artists.append(time_text);
			return artists;

		if(self.get_fps()*self.get_timestep() >= numpy.double(1)):
			raise ValueError(f"timestep \'{timestep}\' is too large")

		ani = FuncAnimation(self.figure, animate, numpy.arange(1, fps*time, dtype=numpy.double),
			init_func=setup, interval=1000 / fps, repeat=False, blit=True);

		# set axis properties
		self.axes.axis('square');
		self.axes.set_xlim(xllim, xhlim);
		self.axes.set_ylim(yllim, yhlim);
		self.axes.set_xlim();
		self.axes.autoscale(False);
		self.axes.axis('off');


		pyplot.show();

	# call Signature:
	## set_simulation_opts(timestep=<timestep>, fps=<fps>, radius=<radius>,
	##       bounds=<bounds>, origin=<[x0, y0]>, trace=<trace>)
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
				else:
					raise IndexError(f"Invalid Key Value \'{key}\'");
			else:
				raise TypeError("key must be of type string");

	def set_timestep(self, timestep):
		self.plot_properties.timestep = numpy.double(timestep);

	def set_fps(self, fps):
		self.plot_properties.fps = numpy.double(fps);

	def set_radius(self, radius):
		self.plot_properties.radius = numpy.double(radius);

	def set_bounds(self, lim):
		self.plot_properties.bounds = numpy.double(lim);

	def set_origin(self, origin):
		self.plot_properties.origin = numpy.array(origin);
		numpy.reshape(self.plot_properties.origin, 2);

	def set_plot_trace(self, trace):
		self.plot_properties.trace_length = numpy.double(trace);

	def set_gravitational_constant(self, Num):
		self.G = numpy.double(Num);

	def get_timestep(self):
		return self.plot_properties.timestep;

	def get_fps(self):
		return self.plot_properties.fps;

	def get_radius(self):
		return self.plot_properties.radius;

	def get_bounds(self):
		return self.plot_properties.bounds;

	def get_origin(self):
		return self.plot_properties.origin;

	def get_gravitational_constant(self):
		return self.G;

	def lock_frame_on(self, planet):
		if(not (Body is type(planet) or Dynamic_Body is type(planet))):
			raise TypeError(f"\'{planet}\' is not a body type")
		self.plot_properties.frame_lockon = planet;

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

	# merge bodies bd1 and bd2 so that momentum is conserved
	def merge_two_bodies(self, bd1, bd2):
		if (bd1 is bd2):
			raise ValueError("Cannot merge the same body onto itself");

		delete_string = '__TO_BE_DELETED__';
		Mass     = bd1.mass + bd2.mass;
		velocity = (1 / Mass) * (bd1.mass * bd1.velocity + \
					bd2.mass * bd2.velocity);

		if (bd2.mass >= bd1.mass):
			if (bd1 is self.plot_properties.frame_lockon):
				self.plot_properties.frame_locon = bd2;
			bd2.mass = Mass;
			bd2.velocity = velocity;
			bd1.name = delete_string;
			bd = bd2;
		else:
			if (bd2 is self.plot_properties.frame_lockon):
				self.plot_properties.frame_locon = bd1;
			bd1.mass = Mass;
			bd1.velocity = velocity;
			bd2.name = delete_string;
			bd = bd1;
		self.remove_body(delete_string);

		return bd;

	# get the radius of a body
	def radius(self, body):
		return body.radius;

	# return true if bd1 and bd2 overlap
	def two_bodies_overlap(self, bd1, bd2):
		norm   = numpy.linalg.norm;

		if (bd1 is bd2):
			raise ValueError("Cannot calculate overlap for the same body");

		if (norm(bd1.position - bd2.position) <= self.radius(bd1) + self.radius(bd2)):
			return True;
		return False;

	# change the inertial frame
	def change_intertial_frame(self, velocity):
		velocity = numpy.array(velocity, dtype=numpy.double);
		numpy.reshape(velocity,2);

		for body in self:
			body.velocity -= velocity;

	# return velocity such that there is no net momentum
	def zero_momentum_frame(self):
		Mass = numpy.double(0);
		velocity = numpy.zeros(2);

		for body in self:
			Mass += body.mass;
			velocity += body.mass * body.velocity;
		return (1/Mass) * velocity;

	# adjust the radius relative to settings
	def adjust_radius(self):
		for body in self:
			if (not body._Costom_radius):
				body.radius = (body.mass / self._min_mass) ** 0.5 * \
					self.plot_properties.radius;

	# get acceleration of body irrespective of referance frame (non-inertial)
	def get_acceleration(self, body):
		G = self.plot_properties.G;
		norm   = numpy.linalg.norm;
		power  = numpy.power;

		# assumes sum calles the appropriate numpy method!!
		return sum((G * bd.mass / power(norm(bd.position - body.position),3)) * \
						(bd.position - body.position)
						for bd in self.every_other_body(body));

	# populate acceleration data with respect to frame of referance
	def populate_acceleration_data(self):
		G = self.plot_properties.G;
		locked_body  = self.plot_properties.frame_lockon;
		norm  = numpy.linalg.norm;

		if (type(locked_body) is NoneType):
			for body in self:
				body.acceleration = self.get_acceleration(body);
		else:
			frame_acceleration = self.get_acceleration(locked_body)
			for body in self.every_other_body(locked_body):
				body.acceleration = self.get_acceleration(body) - frame_acceleration;
			locked_body.acceleration = numpy.array([0,0], dtype=numpy.double);

	# routine for handling collisions
	def _collision_handler(self, body):
		for bd in self.every_other_body(body):
			if(self.two_bodies_overlap(body, bd)):
				self.merge_two_bodies(body,bd);

		if (type(self.plot_properties.frame_lockon) is NoneType):
			self.change_intertial_frame(self.zero_momentum_frame());
		else:
			self.change_intertial_frame(self.plot_properties.frame_lockon.velocity);
			self.plot_properties.frame_lockon.velocity = numpy.array([0,0], dtype=numpy.double);

	# this routine will setup the bodies up for simulation, Specifically:
	## overlapping bodies are merged...
	## the intertial frame is set to the net zero momentum frame
	def setup_bodies(self):
		self.adjust_radius();
		for body in self:
			for bd in self.every_other_body(body):
				if(self.two_bodies_overlap(body,bd)):
					self.merge_two_bodies(body,bd);
		self.populate_acceleration_data();

		if (type(self.plot_properties.frame_lockon) is NoneType):
			self.change_intertial_frame(self.zero_momentum_frame());
		else:
			print(self.plot_properties.frame_lockon);
			print(self.plot_properties.frame_lockon.velocity);
			self.change_intertial_frame(self.plot_properties.frame_lockon.velocity);
			self.plot_properties.frame_lockon.acceleration = numpy.array([0,0], dtype=numpy.double);
			self.plot_properties.frame_lockon.velocity     = numpy.array([0,0], dtype=numpy.double);

	def update_bodies(self, time):
		G = self.plot_properties.G;
		locked_body        = self.plot_properties.frame_lockon;
		norm   = numpy.linalg.norm;
		power  = numpy.power;
		tDelta = self.get_timestep();
		time   = numpy.double(time);

		t = numpy.double(0);
		if (type(locked_body) is NoneType):
			while (t < time):
				for body in self:
					self._collision_handler(body);
				for body in self:
					# assumes sum calles the appropriate numpy method!!
					body.acceleration = sum(
						(G * bd.mass / power(norm(bd.position - body.position),3)) * \
						(bd.position - body.position)
						for bd in self.every_other_body(body));
					body.update_velocity(tDelta);
					body.update_position(tDelta);
				t += tDelta;
		else:
			while (t < time):
				for body in self:
					self._collision_handler(body);
				frame_acceleration = self.get_acceleration(locked_body);
				for body in self.every_other_body(locked_body):
					# assumes sum calles the appropriate numpy method!!
					body.acceleration = sum(
						(G * bd.mass / power(norm(bd.position - body.position),3)) * \
						(bd.position - body.position)
						for bd in self.every_other_body(body)) - frame_acceleration;
					body.update_velocity(tDelta);
					body.update_position(tDelta);
				t += tDelta;

	def plot_setup(self):
		for body in self:
			# prevent previous trace being erased
			if (not hasattr(body,'trace')):
				body.xtrace = [];
				body.ytrace = [];
			body.trace, = self.axes.plot([],[], body.color+'.', lw=1, ms=1);

	def plot_bodies(self):
		artists = [];
		circles  = [];

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
			circles.append(
				pyplot.Circle(body.position, radius=self.radius(body), color=body.color)
				)
			artists.append(body.trace);
		artists.append(self.axes.add_collection(
			matplotlib.collections.PatchCollection(circles, match_original=True)));

		return artists;
