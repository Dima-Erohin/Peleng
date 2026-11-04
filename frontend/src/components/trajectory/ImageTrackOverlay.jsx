import React, { useEffect, useRef, useState } from "react";
import { sampleImageTrack } from "../../trajectory/image-tracks";

export default function ImageTrackOverlay({ track, videoRef, time, ready, statusIndex = 0 }) {
  const [sample, setSample] = useState(null);
  const lastUpdate = useRef(0);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !track || !ready) { setSample(null); return; }
    const update = () => setSample(sampleImageTrack(track, video.currentTime));
    update();
    let callback;
    const onFrame = (now) => {
      if (now - lastUpdate.current >= 75) { lastUpdate.current = now; update(); }
      callback = video.requestVideoFrameCallback(onFrame);
    };
    if (video.requestVideoFrameCallback) callback = video.requestVideoFrameCallback(onFrame);
    video.addEventListener("seeked", update);
    video.addEventListener("timeupdate", update);
    return () => {
      if (callback != null) video.cancelVideoFrameCallback(callback);
      video.removeEventListener("seeked", update);
      video.removeEventListener("timeupdate", update);
    };
  }, [track, ready, videoRef]);
  useEffect(() => {
    if (track && videoRef.current && ready && videoRef.current.paused) {
      setSample(sampleImageTrack(track, videoRef.current.currentTime));
    }
  }, [time, track, ready, videoRef]);
  if (!track || !sample) return null;
  const box = sample.bbox;
  const label = track.label === "drone" ? "Дрон" : track.label || track.object_id;
  return (
    <>
      <svg className="image-track-overlay" viewBox={`0 0 ${track.width} ${track.height}`} preserveAspectRatio="xMidYMid meet" aria-label={`Рамка ${label} в пикселях видео`}>
        {box && <g>
          <rect x={box[0]} y={box[1]} width={box[2]} height={box[3]} fill="none" stroke="#52e1bd" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          <text x={box[0]} y={Math.max(28, box[1] - 12)} fill="#52e1bd" fontSize="30" fontFamily="monospace">{track.object_id}</text>
        </g>}
      </svg>
      <span className={`image-track-status ${box ? "" : "is-lost"}`} style={{ bottom: 6 + statusIndex * 21 }}>
        {box ? `${label} · 2D · NCC ${sample.correlation.toFixed(2)}` : `${label} · нет уверенного совпадения`}
      </span>
    </>
  );
}
