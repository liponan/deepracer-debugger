import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


st.title("🏎️ Deep Racer Debugger")

cols = st.columns(2)

with cols[0]:
    data_track = st.file_uploader("Upload track file", type="npy")

with cols[1]:
    data_log = st.file_uploader("Upload training log", type=["tar", "csv"], accept_multiple_files=True)

if data_track is not None:

    st.header("🏁 Track Analysis")

    track_npy = np.load(data_track)
    # dx = np.diff(0.5 * (data[:-1, 0] + data[1:, 0]))
    dx = 0.5 * (np.diff(track_npy[:-1, 0]) + np.diff(track_npy[1:, 0]))
    dxx = np.diff(track_npy[:, 0], 2)
    dxx = track_npy[2:, 0] + track_npy[0:-2, 0] - 2*track_npy[1:-1, 0]
    # dy = np.diff(0.5 * (data[:-1, 1] + data[1:, 1]))
    dy = 0.5 * (np.diff(track_npy[:-1, 1]) + np.diff(track_npy[1:, 1]))
    dyy = np.diff(track_npy[:, 1], 2)
    k = (dx*dyy - dy*dxx)/(dx**2+dy**2)**(2/3)
    color_code = st.radio("Heat map", ["Mileage", "R", "|R|", "d|R|", "d^2|R|"], horizontal=True)
    show_colorbar = st.toggle("Show color bar")

    with plt.style.context("dark_background"):
        fig = plt.figure()
        plt.plot(track_npy[:, 2], track_npy[:, 3], color="gray")
        plt.plot(track_npy[:, 4], track_npy[:, 5], color="gray")
        if color_code == "Mileage":
            plt.scatter(track_npy[:, 0], track_npy[:, 1], c=np.arange(track_npy.shape[0]), cmap="jet")
        elif color_code == "R":
            vmax = np.max(np.abs(k))
            plt.scatter(track_npy[1:-1, 0], track_npy[1:-1, 1], c=k, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        elif color_code == "|R|":
            plt.scatter(track_npy[1:-1, 0], track_npy[1:-1, 1], c=np.abs(k), cmap="jet")
        elif color_code == "d|R|":
            x = 0.5 * (track_npy[1:-2, 0] + track_npy[2:-1, 0])
            y = 0.5 * (track_npy[1:-2, 1] + track_npy[2:-1, 1])
            dk = np.diff(np.abs(k))
            vmax = np.max(dk)
            plt.scatter(x, y, c=dk, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        elif color_code == "d^2|R|":
            x =track_npy[2:-2, 0]
            y =track_npy[2:-2, 1]
            dk2 = np.diff(np.abs(k), 2)
            vmax = np.max(dk2)
            plt.scatter(x, y, c=dk2, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.axis("off")
        if show_colorbar:
            plt.colorbar()
    st.pyplot(fig)

if len(data_log) > 0:

    st.header("🏆 Training Log")

    df = []
    for f in data_log:
        df.append(pd.read_csv(f))
    df = pd.concat(df, axis=0)
    df = df.drop_duplicates().sort_values(["episode", "steps"])

    episodes = df.episode.unique()
    cols = st.columns(2)
    with cols[0]:
        sel_episode = st.selectbox("Episode", episodes, index=None)
    with cols[1]:
        sel_metric = st.selectbox("Metric", ["steps", "yaw", "steer", "throttle", "reward"])

    with plt.style.context("dark_background"):
        fig = plt.figure()
        if isinstance(track_npy, np.ndarray):
            plt.plot(track_npy[:, 2], track_npy[:, 3], color="gray")
            plt.plot(track_npy[:, 4], track_npy[:, 5], color="gray")
            plt.plot(track_npy[:, 0], track_npy[:, 1], "--", color="w", zorder=-1)
        if sel_episode is not None:
            df_q = df.query(f"episode == {sel_episode}")
            x, y, metics = df_q["X"], df_q["Y"], df_q[sel_metric]
            if sel_metric == "yaw":
                plt.scatter(x, y, c=metics, cmap="hsv", vmin=-180, vmax=180)
            elif sel_metric == "steer":
                plt.scatter(x, y, c=metics, cmap="coolwarm", vmin=-30, vmax=30)
            else:
                plt.scatter(x, y, c=metics, cmap="jet")
        else:
            cmap = cm.viridis
            for i, ep in enumerate(episodes):
                df_q = df.query(f"episode == {ep}")
                x, y = df_q["X"], df_q["Y"]
                plt.plot(x, y, color=cmap(i/len(episodes)))
            

        plt.axis("off")
        st.pyplot(fig)
        st.dataframe(df)