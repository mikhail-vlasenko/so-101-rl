import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { PerceptionScene } from './PerceptionScene.jsx';
import './style.css';
import { SCENE_STYLES } from './sceneStyles.js';
import { DRAFT_DURATION, sampleDraft } from './motionDraft.js';
import { STORY_FRAMES } from './storyFrames.js';
import { FilmScene } from './FilmScene.jsx';
import { FILM_DURATION, FILM_VIDEO, FILM_CAPTION_FILE, sampleFilm } from './filmTimeline.js';

function App() {
  const clean = new URLSearchParams(window.location.search).has('frame');
  const [film, setFilm] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.has('film') || (!params.has('keyframe') && !params.has('style') && !params.has('still') && !params.has('draft'));
  });
  const [keyframe, setKeyframe] = useState(() => {
    const requested = new URLSearchParams(window.location.search).get('keyframe');
    return requested && Object.hasOwn(STORY_FRAMES, requested) ? requested : null;
  });
  const [style, setStyle] = useState(() => {
    if (new URLSearchParams(window.location.search).has('keyframe')) return 'studio';
    const requested = new URLSearchParams(window.location.search).get('style') || 'studio';
    return Object.hasOwn(SCENE_STYLES, requested) ? requested : 'studio';
  });
  const [motion, setMotion] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return !film && !params.has('keyframe') && !params.has('still') && (!params.has('style') || params.get('style') === 'studio');
  });
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const duration = film ? FILM_DURATION : DRAFT_DURATION;
  const timed = film || motion;
  const selectStyle = (value) => {
    setFilm(false);
    setStyle(value);
    setKeyframe(null);
    setMotion(false);
    setPlaying(false);
    const url = new URL(window.location.href);
    url.searchParams.set('style', value);
    url.searchParams.set('still', '');
    url.searchParams.delete('keyframe');
    url.searchParams.delete('film');
    window.history.replaceState(null, '', url);
  };
  const selectKeyframe = (value) => {
    setFilm(false);
    setStyle('studio');
    setMotion(false);
    setPlaying(false);
    setKeyframe(value);
    const url = new URL(window.location.href);
    url.searchParams.set('style', 'studio');
    url.searchParams.set('keyframe', value);
    url.searchParams.delete('still');
    url.searchParams.delete('film');
    window.history.replaceState(null, '', url);
  };
  const openMotion = () => {
    setFilm(false);
    setStyle('studio');
    setKeyframe(null);
    setMotion(true);
    setPlaying(false);
    const url = new URL(window.location.href);
    url.searchParams.set('style', 'studio');
    url.searchParams.delete('still');
    url.searchParams.delete('keyframe');
    url.searchParams.delete('film');
    window.history.replaceState(null, '', url);
    window.dispatchEvent(new Event('perception-reset-camera'));
  };
  const openFilm = () => {
    setFilm(true);
    setStyle('studio');
    setKeyframe(null);
    setMotion(false);
    setPlaying(false);
    setTime(0);
    const url = new URL(window.location.href);
    url.searchParams.set('film', '');
    for (const parameter of ['style', 'keyframe', 'still', 'draft']) url.searchParams.delete(parameter);
    window.history.replaceState(null, '', url);
    window.dispatchEvent(new Event('perception-reset-camera'));
  };
  return (
    <main className={clean ? 'clean' : 'workspace'}>
      {!clean && <header>
        <div><strong>Perception</strong><span>{film ? `Full film · ${FILM_DURATION} seconds` : keyframe ? 'Storyboard · static keyframes' : motion ? 'Motion draft · 14 seconds' : '3D scene study'}</span></div>
        <nav>
          <button aria-pressed={film} onClick={openFilm}>Full video</button>
          <button aria-pressed={Boolean(keyframe)} onClick={() => selectKeyframe(keyframe || 'views')}>Storyboard frames</button>
          <button aria-pressed={motion} onClick={openMotion}>Motion draft</button>
          <button aria-pressed={!timed && !keyframe} onClick={() => selectStyle('studio')}>Scene styles</button>
          <button onClick={() => window.dispatchEvent(new Event('perception-reset-camera'))}>Reset view</button>
          {keyframe === 'surface' && <button onClick={() => window.dispatchEvent(new Event('perception-replay-orbit'))}>Replay orbit</button>}
          <button onClick={() => window.dispatchEvent(new Event('perception-export-frame'))}>Export PNG ↗</button>
          {timed && <a className="download" href={film ? FILM_VIDEO : '/renders/perception-motion-v2.mp4'} download>{film ? 'Last exported MP4 ↗' : 'Download MP4 ↗'}</a>}
        </nav>
      </header>}
      {!clean && keyframe && <div className="frame-picker" aria-label="Storyboard frames">
        {Object.entries(STORY_FRAMES).map(([id, frame]) => <button key={id} aria-pressed={keyframe === id} onClick={() => selectKeyframe(id)}>
          <img src={`/keyframes/story-${id}.png`} alt="" />
          <span>{frame.title}</span>
        </button>)}
      </div>}
      {!clean && !timed && !keyframe && <div className="style-picker" aria-label="Scene style">
        {Object.entries(SCENE_STYLES).map(([id, theme]) => <button key={id} aria-pressed={style === id} onClick={() => selectStyle(id)}>{theme.title}<small>{theme.description}</small></button>)}
      </div>}
      <div className="stage">{film ? <FilmScene onTime={setTime} onPlaying={setPlaying} /> : <PerceptionScene style={style} motion={motion} keyframe={keyframe} onTime={setTime} onPlaying={setPlaying} />}</div>
      {!clean && timed && <div className="playback">
        <button onClick={() => window.dispatchEvent(new Event(playing ? 'perception-pause' : 'perception-play'))}>{playing ? 'Pause' : time >= duration ? 'Replay' : 'Play'}</button>
        <input aria-label="Timeline" type="range" min="0" max={duration} step="0.01" value={time} onChange={(event) => window.dispatchEvent(new CustomEvent('perception-seek', { detail: Number(event.target.value) }))} />
        <output>{time.toFixed(1)} / {duration}s</output>
      </div>}
      {!clean && <footer><span>{film ? sampleFilm(time).chapter.title : keyframe ? STORY_FRAMES[keyframe].note : motion ? sampleDraft(time).phase : 'Drag to orbit · scroll to zoom · right-drag to pan'}</span><span>{timed ? 'Drag to pause and explore · play resumes the authored camera' : 'Drag to orbit · scroll to zoom · right-drag to pan'}</span></footer>}
      {!clean && film && <details className="film-notes">
        <summary>How this maps to the real system</summary>
        <p>SAM 3 finds the object from the prompt “sponge”; SAM 2 tracks its pixels in each image. The two mask centers provide an approximate position. That estimate can also have timing errors.</p>
        <p>Dense stereo matches image details to measure the visible surface. Independent cameras expose at different times: rectifying their images aligns geometry, but cannot align the moments they captured. During movement the system holds its last accepted surface measurement, then attempts a refresh after the object settles.</p>
        <p>This film uses rendered illustrations of the 6 × 4 × 2.5 cm sponge and a fixed camera pair. Motion eases into and out of a steady cruising speed; travel is compressed while the sponge is faded out for the depth explanation. The spacing between shutters is illustrative, not a measured camera delay. The cloud shows only the three observed faces.</p>
        <a href={FILM_CAPTION_FILE} download>Last exported captions</a>
      </details>}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
