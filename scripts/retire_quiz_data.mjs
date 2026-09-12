import fs from 'node:fs';
const match=s=>/quiz|blind[ -]?test/i.test(s);
const clean=s=>s.replace(/, quiz et/gi,' et').replace(/, Quiz et/g,' et').replace(/, quiz,/gi,',').replace(/six activit/gi,'cinq activit');
const p='src/data/knowledge_base.json';const kb=JSON.parse(fs.readFileSync(p));
kb.global.routing=kb.global.routing.map(clean);
for(const id of ['retroworld','runningman']){const b=kb.brands[id];b.summary=clean(b.summary);for(const k of ['highlights','booking_links','quick_actions','knowledge_cards','offers']) b[k]=(b[k]||[]).filter(x=>!match(typeof x==='string'?x:JSON.stringify(x)));}
fs.writeFileSync(p,JSON.stringify(kb,null,2)+'\n');
for(const id of ['retroworld','runningman']){const file=`static/faq_${id}.json`;const f=JSON.parse(fs.readFileSync(file));f.items=f.items.filter(x=>!match(x.question||'')).map(x=>({...x,answer:clean(x.answer||''),tags:(x.tags||[]).filter(x=>!match(x))})).filter(x=>!match(x.answer));fs.writeFileSync(file,JSON.stringify(f,null,2)+'\n');}
const widget='static/chat-widget.html';fs.writeFileSync(widget,clean(fs.readFileSync(widget,'utf8')));
