import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import gym


def get_cart_location(env, screen_width=600):
    """
    Get the position of cart
    :param env: environment, in this file means cartpole-v0
    :param screen_width:screen width defined in gym
    :return:middle position of cart
    """
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen(env, use_pytorch=True):
    if use_pytorch:  # in pytorch shapes are (C, H, W)
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = screen[:, 160:320, :]
    else:  # in tensorflow shapes are (H, W, C)
        screen = env.render(mode='rgb_array')
        screen = screen[160:320, :, :]
    view_width = 320
    cart_location = get_cart_location(env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (view_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    if use_pytorch:  # get width
        screen = screen[:, :, slice_range]
    else:
        screen = screen[:, slice_range, :]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return screen


env = gym.make('CartPole-v0')
env = env.unwrapped
env.reset()
last_screen = get_screen(env)
env.step(1)
current_screen = get_screen(env)
state = current_screen - last_screen
print(state)
plt.figure()
plt.imshow(state.transpose(1,2,0), interpolation=None)
plt.show()


