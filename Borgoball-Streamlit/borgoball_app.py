import streamlit as st
import numpy as np
from math import cos, sin, radians
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io

def create_baseball_field():
    """Create baseball field layout for 3D plot"""
    # Field dimensions (in feet)
    pitching_rubber_to_home = 60.5  # Distance for youth baseball
    base_distance = 45  # Reduced base path length

    # Create infield diamond
    diamond_x = [0, base_distance, base_distance, 0, 0]  # Home to 1st to 2nd to 3rd to Home
    diamond_y = [0, 0, base_distance, base_distance, 0]
    diamond_z = [0, 0, 0, 0, 0]  # All points at ground level

    # Create pitcher's mound
    mound_x = [pitching_rubber_to_home]
    mound_y = [base_distance/2]
    mound_z = [1]  # Slight elevation for mound

    # Create outfield fence
    num_points = 50  # Number of points for smooth curve
    t = np.linspace(0, 1, num_points)

    # Create the fence points using quadratic Bézier curve
    # Control points for the curve: (0,100), (90,90), (100,0)
    p0_x, p0_y = 0, 100    # Start point (third base line)
    p1_x, p1_y = 90, 90    # Control point (center field)
    p2_x, p2_y = 100, 0    # End point (first base line)

    # Quadratic Bézier curve formula for curved section only
    fence_x = (1-t)**2 * p0_x + 2*(1-t)*t * p1_x + t**2 * p2_x
    fence_y = (1-t)**2 * p0_y + 2*(1-t)*t * p1_y + t**2 * p2_y

    # Create ground level and top level points for the curved section
    ground_x = fence_x
    ground_y = fence_y
    ground_z = np.zeros_like(fence_x)

    top_x = fence_x
    top_y = fence_y
    top_z = np.ones_like(fence_x) * 8  # 8-foot high fence

    # Stack vertical lines at each point of the curve
    vertical_lines_x = []
    vertical_lines_y = []
    vertical_lines_z = []

    for i in range(len(ground_x)):
        vertical_lines_x.extend([ground_x[i], ground_x[i]])
        vertical_lines_y.extend([ground_y[i], ground_y[i]])
        vertical_lines_z.extend([ground_z[i], top_z[i]])

    # Combine all points
    fence_x = np.array(vertical_lines_x)
    fence_y = np.array(vertical_lines_y)
    fence_z = np.array(vertical_lines_z)

    return {
        'diamond': (diamond_x, diamond_y, diamond_z),
        'mound': (mound_x, mound_y, mound_z),
        'fence': (fence_x, fence_y, fence_z)
    }

def calculate_trajectory_points_3d(exit_velocity, launch_angle, direction_angle, num_points=100):
    """
    Calculate points along the trajectory path for 3D visualization.
    Adjusted for 1-ounce foam baseball characteristics including bouncing and rolling.
    """
    # Convert inputs
    velocity = exit_velocity * 1.467  # mph to ft/s
    theta = radians(launch_angle)
    phi = radians(90 - direction_angle)  # Convert direction angle to radians

    # Initial velocities in 3D
    vx = velocity * cos(theta) * cos(phi)
    vy = velocity * sin(theta)
    vz = velocity * cos(theta) * sin(phi)

    # Constants
    g = 32.174  # Gravity (ft/s²)
    mass = 1/16  # 1 ounce = 1/16 pound
    air_resistance_factor = 0.45  # Increased air resistance for foam ball
    coefficient_of_restitution = 0.3  # Foam ball bounces less than regular baseball
    rolling_friction = 0.95  # Rolling friction coefficient (higher means more friction)

    # Lists to store all trajectory points
    all_x = []
    all_y = []
    all_z = []

    # Initial position
    x = 0
    y = 0.1  # Start slightly above ground
    z = 0

    t = 0
    dt = 0.05  # Time step
    min_bounce_velocity = 0.5  # Minimum velocity for bounce to occur
    min_movement_velocity = 0.1  # Minimum velocity to continue motion

    # Track if ball is rolling and first ground contact
    is_rolling = False
    first_ground_contact = None
    first_ground_distance = None

    while len(all_x) < num_points:
        # Update position
        x += vx * dt * air_resistance_factor
        y += vy * dt if not is_rolling else 0
        z += vz * dt * air_resistance_factor

        # Update velocity due to gravity if not rolling
        if not is_rolling:
            vy -= g * dt

        # Ground collision
        if y < 0.1:  # Ground level
            y = 0.1  # Keep slightly above ground

            # Record first ground contact
            if first_ground_contact is None:
                first_ground_contact = (x, y, z)
                first_ground_distance = np.sqrt(x**2 + z**2)

            # Calculate total horizontal velocity
            horizontal_velocity = np.sqrt(vx**2 + vz**2)

            if not is_rolling and abs(vy) > min_bounce_velocity:
                # Bounce with energy loss
                vy = -vy * coefficient_of_restitution
                # Reduce horizontal velocity due to impact friction
                vx *= 0.8
                vz *= 0.8
            else:
                # Start rolling
                is_rolling = True
                vy = 0
                # Apply rolling friction to horizontal velocities
                vx *= rolling_friction
                vz *= rolling_friction

        # Store points
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

        # Stop if the ball has essentially stopped moving
        if is_rolling and np.sqrt(vx**2 + vz**2) < min_movement_velocity:
            # Add final resting position a few more times to show it clearly
            for _ in range(5):
                all_x.append(x)
                all_y.append(y)
                all_z.append(z)
            break

        t += dt

    return np.array(all_x), np.array(all_y), np.array(all_z), first_ground_distance

def plot_trajectory_3d(x_points, y_points, z_points):
    """Create an interactive 3D plot of the ball's trajectory with baseball field and animation"""
    fig = go.Figure()

    # Add baseball field elements
    field = create_baseball_field()

    # Add ground plane first
    fig.add_trace(go.Surface(
        x=np.linspace(-10, 200, 2),
        y=np.linspace(-10, 200, 2),
        z=np.zeros((2, 2)),
        colorscale=[[0, 'green'], [1, 'green']],
        showscale=False,
        opacity=0.3,
        name='Field',
        showlegend=False
    ))

    # Add outfield fence
    fig.add_trace(go.Scatter3d(
        x=field['fence'][0],
        y=field['fence'][1],
        z=field['fence'][2],
        mode='lines',
        name='Outfield Fence',
        line=dict(color='blue', width=3),
        showlegend=False
    ))

    # Add bases as markers
    bases_x = [0, 45, 45, 0]  # Home, 1st, 2nd, 3rd
    bases_y = [0, 0, 45, 45]
    bases_z = [0.1, 0.1, 0.1, 0.1]  # Slightly above ground

    # Create frames for animation
    frames = []
    for i in range(len(x_points)):
        frame = go.Frame(
            data=[
                # Keep ground plane in each frame
                go.Surface(
                    x=np.linspace(-10, 200, 2),
                    y=np.linspace(-10, 200, 2),
                    z=np.zeros((2, 2)),
                    colorscale=[[0, 'green'], [1, 'green']],
                    showscale=False,
                    opacity=0.3,
                    name='Field',
                    showlegend=False
                ),
                # Keep infield in each frame
                go.Scatter3d(
                    x=field['diamond'][0],
                    y=field['diamond'][1],
                    z=field['diamond'][2],
                    mode='lines',
                    name='Infield',
                    line=dict(color='brown', width=3),
                    showlegend=False
                ),
                # Keep bases in each frame
                go.Scatter3d(
                    x=bases_x,
                    y=bases_y,
                    z=bases_z,
                    mode='markers',
                    name='Bases',
                    marker=dict(size=8, color='white', symbol='square'),
                    showlegend=False
                ),
                # Keep outfield fence in each frame
                go.Scatter3d(
                    x=field['fence'][0],
                    y=field['fence'][1],
                    z=field['fence'][2],
                    mode='lines',
                    name='Fence',
                    line=dict(color='blue', width=3),
                    showlegend=False
                ),
                # Trailing path
                go.Scatter3d(
                    x=x_points[:i+1],
                    y=z_points[:i+1],
                    z=y_points[:i+1],
                    mode='lines',
                    name='Ball Path',
                    line=dict(color='rgba(255, 0, 0, 0.7)', width=3),
                    showlegend=False
                ),
                # Current ball position
                go.Scatter3d(
                    x=[x_points[i]],
                    y=[z_points[i]],
                    z=[y_points[i]],
                    mode='markers',
                    name='Ball',
                    marker=dict(
                        size=12,
                        color='white',
                        symbol='circle',
                        line=dict(color='red', width=2)
                    ),
                    showlegend=False
                )
            ],
            name=f"frame{i}"
        )
        frames.append(frame)

    # Add initial static display
    fig.add_trace(go.Scatter3d(
        x=field['diamond'][0],
        y=field['diamond'][1],
        z=field['diamond'][2],
        mode='lines',
        name='Infield',
        line=dict(color='brown', width=3),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=bases_x,
        y=bases_y,
        z=bases_z,
        mode='markers',
        name='Bases',
        marker=dict(size=8, color='white', symbol='square'),
        showlegend=False
    ))

    # Add initial ball position
    fig.add_trace(go.Scatter3d(
        x=[x_points[0]],
        y=[z_points[0]],
        z=[y_points[0]],
        mode='markers',
        name='Ball',
        marker=dict(
            size=12,
            color='white',
            symbol='circle',
            line=dict(color='red', width=2)
        ),
        showlegend=False
    ))

    # Update animation settings for smoother bouncing visualization
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 30, 'redraw': True},  # Faster animation
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                        'mode': 'immediate'
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'x': 0.1,
            'y': 0
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(k),
                    'method': 'animate'
                } for k, f in enumerate(frames)
            ]
        }]
    )

    # Update layout
    fig.update_layout(
        title='3D Borgoball Trajectory with Bounces',
        scene=dict(
            xaxis_title='Distance (feet)',
            yaxis_title='Field Position (feet)',
            zaxis_title='Height (feet)',
            zaxis=dict(range=[0, 85]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4)
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Add the frames to the figure
    fig.frames = frames

    return fig

def calculate_distance(exit_velocity, launch_angle):
    """
    Calculate the distance traveled by a 1-ounce foam baseball using simplified projectile motion equations.
    Adjusted for foam baseball characteristics including higher air resistance and lower mass.
    """
    # Convert velocity to ft/s
    velocity = exit_velocity * 1.467
    # Convert angle to radians
    theta = radians(launch_angle)
    # Initial velocities
    vx = velocity * cos(theta)
    vy = velocity * sin(theta)
    # Gravity (ft/s²)
    g = 32.174
    # Time of flight
    t = 2 * vy / g
    # Air resistance factor for 1-ounce foam ball
    air_resistance_factor = 0.45
    # Horizontal distance with increased air resistance factor for foam ball
    distance = vx * t * air_resistance_factor
    return distance

def validate_inputs(exit_velocity, launch_angle):
    """Validate input parameters and return error message if invalid."""
    # Lower max velocity for foam baseball
    if exit_velocity < 0 or exit_velocity > 70:
        return "Exit velocity must be between 0 and 70 mph for a Borgoball"
    if launch_angle < 0 or launch_angle > 90:
        return "Launch angle must be between 0 and 90 degrees"
    return None

# Main UI section
st.title("🎯 Borgoball Trajectory Calculator")

st.markdown("""
This calculator estimates the distance a Borgoball (1-ounce foam baseball) will travel based on exit velocity and launch angle.
The calculation is adjusted for Borgoball characteristics including higher air resistance and lower mass.

**Instructions:**
1. Enter the exit velocity (0-70 mph)
2. Enter the launch angle (0-90 degrees)
3. Adjust the direction angle using the slider
4. Click 'Calculate' to see the results
""")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    exit_velocity = st.number_input(
        "Exit Velocity (mph)",
        min_value=0.0,
        max_value=70.0,
        value=40.0,
        step=0.1,
        help="The speed of the Borgoball when it leaves the bat (0-70 mph)"
    )

with col2:
    launch_angle = st.number_input(
        "Launch Angle (degrees)",
        min_value=0.0,
        max_value=90.0,
        value=30.0,
        step=0.1,
        help="The vertical angle of the ball's trajectory (0-90 degrees)"
    )

# Direction angle slider
direction_angle = st.slider(
    "Hit Direction",
    min_value=0.0,
    max_value=90.0,
    value=45.0,
    step=1.0,
    help="0° is along third base line, 90° is along first base line"
)

# Add clarifying caption text
st.caption("0° = Left Field Line (3rd Base) • 45° = Dead Center Field • 90° = Right Field Line (1st Base)")

if st.button("Calculate"):
    # Validate inputs
    error = validate_inputs(exit_velocity, launch_angle)

    if error:
        st.error(error)
    else:
        # Calculate trajectory with selected direction
        x_points, y_points, z_points, first_ground_distance = calculate_trajectory_points_3d(
            exit_velocity, launch_angle, direction_angle
        )

        # Calculate total distance
        total_distance = np.sqrt(
            (x_points[-1] - x_points[0])**2 +
            (z_points[-1] - z_points[0])**2 +
            (y_points[-1] - y_points[0])**2
        )

        # Display results with both distances
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Initial Carry Distance", f"{first_ground_distance:.1f} feet")
        with col2:
            st.metric("Total Distance (with roll)", f"{total_distance:.1f} feet")

        # Generate and display 3D trajectory plot
        fig_3d = plot_trajectory_3d(x_points, y_points, z_points)
        st.plotly_chart(fig_3d, use_container_width=True)

        # Add explanation of 3D view
        st.info("""
        **3D View Controls:**
        - Rotate: Click and drag
        - Zoom: Scroll or pinch
        - Pan: Right-click and drag
        - Reset: Double-click
        """)

        # Add context adjusted for foam baseball
        if total_distance < 100:
            st.info("This is a short hit, typical for a soft contact or bunt with a Borgoball.")
        elif total_distance < 150:
            st.info("This is a medium-length hit for a Borgoball.")
        elif total_distance < 200:
            st.info("This is a long hit with a Borgoball - great contact!")
        else:
            st.info("That's an incredible hit with a Borgoball! Maximum power!")

# Add footer with additional information
st.markdown("""
---
**Note:** This calculator:
- Is specifically calibrated for Borgoballs
- Uses simplified projectile motion equations
- Accounts for the higher air resistance of Borgoballs
- Assumes sea-level conditions

The results are approximate and may vary based on actual conditions like wind, humidity, and altitude.
""")
