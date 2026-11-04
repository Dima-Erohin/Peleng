import React, { useEffect, useState } from "react";
import { frameBoxes } from "../../trajectory/observations.js";

export default function TruckVideoOverlay({ point, cameraIndex, videoRef, ready, fps, width, height, trajectoryMatches }) {
  const [videoTime, setVideoTime] = useState(-1);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !ready) { setVideoTime(-1); return; }
    const update = () => setVideoTime(video.currentTime);
    update();
    let callback;
    const onFrame = () => { update(); callback = video.requestVideoFrameCallback(onFrame); };
    if (video.requestVideoFrameCallback) callback = video.requestVideoFrameCallback(onFrame);
    video.addEventListener("seeked", update);
    video.addEventListener("timeupdate", update);
    return () => {
      if (callback != null) video.cancelVideoFrameCallback(callback);
      video.removeEventListener("seeked", update);
      video.removeEventListener("timeupdate", update);
    };
  }, [ready, videoRef, cameraIndex, point]);
  const boxes = ready ? frameBoxes(point, cameraIndex, videoTime, fps) : [];
  return <>
    <svg className="image-track-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-label="Детекции грузовика на обработанном кадре">
      {boxes.map((box, index) => {
        const detectionIndex = point.dets[cameraIndex].indexOf(box);
        const used = trajectoryMatches && point.used_det?.some(([camera, detection]) => camera === cameraIndex && detection === detectionIndex);
        const color = used ? "#52e1bd" : "#dfbd78";
        return <g key={index}>
          <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} fill="none" stroke={color}
            strokeWidth="2" strokeDasharray={used ? undefined : "6 4"} vectorEffect="non-scaling-stroke" />
          <text x={box[0]} y={Math.max(28, box[1] - 12)} fill={color} fontSize="28" fontFamily="TikTok Sans, sans-serif">
            {`truck · ${Math.round(box[4] * 100)}% · ${!trajectoryMatches ? "детекция" : used ? "XYZ" : "кандидат"}`}
          </text>
        </g>;
      })}
    </svg>
    <span className={`image-track-status ${boxes.length ? "" : "is-lost"}`} style={{ bottom: 6 }}>
      {boxes.length ? `Грузовик · кадр ${point.frame} · ${boxes.length} дет.` : "Грузовик · нет детекции на показанном кадре"}
    </span>
  </>;
}
