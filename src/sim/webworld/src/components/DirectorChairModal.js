import React, { useEffect, useState } from 'react';
import './DirectorChairModal.css';

const QUESTION_PARTS = [['stakes', 'At stake'], ['tension', 'What pulls against what'],
                        ['success', 'Success'], ['failure', 'Failure']];
const FRAME_PARTS = [['settles', 'Settles'], ['open', 'Leaves open'], ['last', 'Ends on']];

function Field({ label, children }) {
  if (!children || (Array.isArray(children) && children.length === 0)) return null;
  return (
    <div className="plan-field">
      <span className="plan-label">{label}</span>
      <span className="plan-value">{children}</span>
    </div>
  );
}

function Scene({ scene, now }) {
  const playing = !!now;
  const status = playing ? 'playing' : scene.played ? 'played' : 'planned';
  const happened = (scene.outcomes || []).filter(o => o.happened);
  return (
    <div className={`plan-scene ${status}`}>
      <div className="plan-scene-title">
        <span>Scene {scene.number}: {scene.title}</span>
        <span className={`plan-status ${status}`}>{status}</span>
      </div>
      <Field label="Where">{[scene.place, scene.when].filter(Boolean).join(', ')}</Field>
      {(scene.cast || []).map(c => (
        <Field key={c.name} label={c.name}>
          {c.goal}{c.keeps && c.keeps.length > 0 ? ` (keeps back: ${c.keeps.join('; ')})` : ''}
        </Field>
      ))}
      <Field label="Opens">{scene.setup}</Field>
      <Field label="Planned ending">{scene.ending}</Field>
      {playing && now.outcomes.length > 0 && (
        <div className="plan-field">
          <span className="plan-label">Ending as outcomes</span>
          <ul className="plan-list">
            {now.outcomes.map((o, i) => (
              <li key={i}><span className={`plan-status ${o.state}`}>{o.state}</span> {o.text}</li>
            ))}
          </ul>
        </div>
      )}
      {playing && now.beats.length > 0 && (
        <div className="plan-field">
          <span className="plan-label">Beats</span>
          <ol className="plan-list">
            {now.beats.map((b, i) => (
              <li key={i} className={b.played ? 'plan-done' : b.playing ? 'plan-current' : ''}>{b.who}: {b.what}</li>
            ))}
          </ol>
        </div>
      )}
      {scene.played && (
        <Field label="What happened">
          {happened.length > 0 ? happened.map(o => o.text).join('; ') : scene.wrapup}
        </Field>
      )}
    </div>
  );
}

function Act({ act, now }) {
  const playing = now && now.act === act.number;
  const status = act.played ? 'played' : playing ? 'playing' : act.written ? 'planned' : 'being written';
  return (
    <div className="plan-act">
      <div className="plan-act-title">
        <span>Act {act.number}{act.written ? `: ${act.title}` : ''} <em>({act.kind})</em></span>
        <span className={`plan-status ${status.replace(' ', '-')}`}>{status}</span>
      </div>
      <Field label="Focus">{act.focus}</Field>
      <Field label="At stake">{act.stakes}</Field>
      {Object.entries(act.roles || {}).map(([name, role]) => <Field key={name} label={name}>{role}</Field>)}
      <Field label="Opens">{act.opens}</Field>
      <Field label="Planned ending">{act.ending}</Field>
      {(act.scenes || []).map((scene, i) => (
        <Scene key={i} scene={scene}
               now={playing && now.scene === scene.number && now.title === scene.title ? now : null} />
      ))}
      {(act.choices || []).map((c, i) => (
        <Field key={i} label={`${c.character}'s choice`}>{c.decision} — {c.choice} (in '{c.scene}')</Field>
      ))}
      <Field label="What it came to">{act.summary}</Field>
    </div>
  );
}

function Plan({ plan }) {
  if (!plan) return <div className="plan-empty">No plan yet: load a play and run it.</div>;
  const q = plan.question || {};
  const frame = plan.frame || {};
  const acts = plan.acts || [];
  const settled = q.text || q.question || acts.length > 0 || (plan.facts || []).length > 0;
  if (!settled) return <div className="plan-empty">Nothing is settled yet: the plan is written when the play is run.</div>;
  const toCome = (plan.act_count || 0) + 1 - acts.length;
  return (
    <div className="plan">
      <div className="plan-section">
        <h4>The question</h4>
        <div className="plan-text">{q.question || q.text}</div>
        {QUESTION_PARTS.map(([key, label]) => <Field key={key} label={label}>{q[key]}</Field>)}
        {Object.entries(q.roles || {}).map(([name, role]) => <Field key={name} label={name}>{role}</Field>)}
      </div>
      {FRAME_PARTS.some(([key]) => frame[key]) && (
        <div className="plan-section">
          <h4>This episode</h4>
          {FRAME_PARTS.map(([key, label]) => <Field key={key} label={label}>{frame[key]}</Field>)}
        </div>
      )}
      {(plan.facts || []).length > 0 && (
        <div className="plan-section">
          <h4>What lies behind the play</h4>
          <ul className="plan-list">
            {plan.facts.map((f, i) => (
              <li key={i}>
                {f.text} <em>({f.of ? `${f.of}'s own past; ` : ''}known to: {(f.known_to || []).join(', ') || 'nobody'}
                {f.open ? '; behind the open question' : ''})</em>
              </li>
            ))}
          </ul>
        </div>
      )}
      {(plan.clues || []).length > 0 && (
        <div className="plan-section">
          <h4>Clues placed</h4>
          <ul className="plan-list">
            {plan.clues.map((c, i) => (
              <li key={i}>
                {c.thing}, at {c.place}{c.hidden ? ' (hidden)' : ''}: {c.touches}
                {c.says ? <div className="plan-quote">It reads: {c.says}</div> : null}
              </li>
            ))}
          </ul>
        </div>
      )}
      {(plan.threads || []).length > 0 && (
        <div className="plan-section">
          <h4>Planted in play so far</h4>
          <ul className="plan-list">
            {plan.threads.map((t, i) => (
              <li key={i}>{t.kind === 'found' ? `found by ${t.who}` : `told by ${t.who}`}: {t.what} <em>(in '{t.scene}')</em></li>
            ))}
          </ul>
        </div>
      )}
      <div className="plan-section">
        <h4>Acts</h4>
        {acts.map(act => <Act key={act.number} act={act} now={plan.now} />)}
        {plan.act_count > 0 && toCome > 0 && (
          <div className="plan-empty">
            {toCome} more {toCome === 1 ? 'act' : 'acts'} to come (the last is the closing act); each is written when the one before it has been played.
          </div>
        )}
        {acts.length === 0 && <div className="plan-empty">No act has been written yet.</div>}
      </div>
    </div>
  );
}

function DirectorChairModal({ characters, plan, onClose, sendCommand }) {
  const controlTypes = ['Act', 'Scene', 'Goal', 'Task', 'Action'];
  const [tab, setTab] = useState('controls');

  // the plan is sent whenever it changes; ask once on opening, for a page loaded after the last change
  useEffect(() => { sendCommand('get_plan'); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="director-chair-modal">
      <div className="director-chair-content">
        <div className="director-chair-header">
          <h3>Director's Chair</h3>
          <div>
            {tab === 'controls' && <button onClick={() => {
              const autonomySettings = {};
              Object.values(characters).forEach(character => {
                autonomySettings[character.name] = {
                  act: document.querySelector(`[data-character="${character.name}"][data-control="Act"]`).classList.contains('automatic'),
                  scene: document.querySelector(`[data-character="${character.name}"][data-control="Scene"]`).classList.contains('automatic'),
                  goal: document.querySelector(`[data-character="${character.name}"][data-control="Goal"]`).classList.contains('automatic'),
                  task: document.querySelector(`[data-character="${character.name}"][data-control="Task"]`).classList.contains('automatic'),
                  action: document.querySelector(`[data-character="${character.name}"][data-control="Action"]`).classList.contains('automatic')
                };
              });
              sendCommand('set_autonomy', { autonomy: autonomySettings });
              onClose();
            }}>Save</button>}
            {tab === 'plan' && <button onClick={() => sendCommand('get_plan')}>Refresh</button>}
            <button onClick={onClose}>×</button>
          </div>
        </div>

        <div className="director-chair-tabs">
          <button className={tab === 'controls' ? 'active' : ''} onClick={() => setTab('controls')}>Controls</button>
          <button className={tab === 'plan' ? 'active' : ''} onClick={() => setTab('plan')}>Plan</button>
        </div>

        {/* kept mounted while the plan is shown: the toggles hold their own state */}
        <div className="character-columns" style={{ display: tab === 'controls' ? undefined : 'none' }}>
          {Object.values(characters).map(character => (
            <div key={character.name} className="character-column">
              <div className="character-name">{character.name}</div>
              <div className="control-buttons">
                {controlTypes.map(controlType => (
                  <div key={controlType} className="control-row">
                    <span className="control-label">{controlType}</span>
                    <button
                      className="control-toggle automatic"
                      data-character={character.name}
                      data-control={controlType}
                      onClick={(e) => {
                        e.target.classList.toggle('automatic');
                        e.target.classList.toggle('manual');
                        e.target.textContent = e.target.classList.contains('automatic') ? 'Autonomous' : 'Manual';
                      }}
                    >
                      Autonomous
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {tab === 'plan' && <Plan plan={plan} />}
      </div>
    </div>
  );
}

export default DirectorChairModal;
