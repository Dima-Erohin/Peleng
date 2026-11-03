import React, { lazy, Suspense, useState } from "react";
import TrajectoryWorkspace from "./components/trajectory/TrajectoryWorkspace";
import "./trajectory.css";

const LiveViewer = lazy(() => import("./viewer.jsx"));

export default function App() {
  const initialLive = new URLSearchParams(location.search).get("workspace") === "live";
  const [workspace, setWorkspace] = useState(initialLive ? "live" : "trajectories");
  const [liveVisited, setLiveVisited] = useState(initialLive);
  const [trajectoryVisited, setTrajectoryVisited] = useState(!initialLive);
  const switchWorkspace = (value) => {
    setWorkspace(value);
    if (value === "live") setLiveVisited(true);
    else setTrajectoryVisited(true);
    const url = new URL(location.href);
    url.searchParams.set("workspace", value);
    history.replaceState(null, "", url);
  };
  const navigation = (
    <nav className="workspace-tabs" aria-label="Режим работы">
      <button type="button" aria-current={workspace === "trajectories" ? "page" : undefined} onClick={() => switchWorkspace("trajectories")}>Траектории</button>
      <button type="button" aria-current={workspace === "live" ? "page" : undefined} onClick={() => switchWorkspace("live")}>Live-камеры</button>
    </nav>
  );
  return (
    <>
      {trajectoryVisited && <div hidden={workspace !== "trajectories"}><TrajectoryWorkspace active={workspace === "trajectories"} navigation={navigation} /></div>}
      {liveVisited && <div hidden={workspace !== "live"}><Suspense fallback={<p role="status">Загружаем live-интерфейс…</p>}><LiveViewer navigation={navigation} /></Suspense></div>}
    </>
  );
}
