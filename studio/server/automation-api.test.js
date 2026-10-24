import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import { createStore } from './store.js'
import { studioApi } from './api.js'

let root, server, base, starts, cancelled, store
const job={id:'10000000-0000-0000-0000-000000000000',status:'running',items:[]}
beforeEach(async()=>{
 root=mkdtempSync(path.join(os.tmpdir(),'studio-automation-api-'));starts=[];cancelled=[]
 const automation={summary:()=>({active:job,history:[job]}),get:id=>id===job.id?job:null,
  start:options=>{starts.push(options);return job},cancel:id=>{cancelled.push(id);return {...job,cancelRequested:true}}}
 let handler
 store=createStore(root)
 studioApi({store,queue:{state:()=>({running:null,queued:[]})},automation})
  .configureServer({middlewares:{use:f=>{handler=f}}})
 server=http.createServer((req,res)=>handler(req,res,()=>{res.statusCode=404;res.end()}))
 await new Promise(r=>server.listen(0,'127.0.0.1',r));base=`http://127.0.0.1:${server.address().port}`
})
afterEach(async()=>{await new Promise(r=>server.close(r));rmSync(root,{recursive:true,force:true})})
test('batch routes preserve options and expose persisted progress/cancellation',async()=>{
 const summary=await(await fetch(base+'/api/automation')).json();expect(summary.active.id).toBe(job.id)
 const options={slugs:['barbell-bent-over-row'],checkOnly:true,forceRebuild:false}
 const created=await fetch(base+'/api/automation',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(options)})
 expect(created.status).toBe(201);expect(starts).toEqual([options])
 expect((await(await fetch(base+'/api/automation/'+job.id)).json()).id).toBe(job.id)
 const stopped=await(await fetch(base+'/api/automation/'+job.id+'/cancel',{method:'POST'})).json()
 expect(stopped.cancelRequested).toBe(true);expect(cancelled).toEqual([job.id])
})
test('a foreign page cannot start or cancel expensive local batches',async()=>{
 for(const url of ['/api/automation','/api/automation/'+job.id+'/cancel']){
  expect((await fetch(base+url,{method:'POST',headers:{origin:'https://foreign.example'},body:'{}'})).status).toBe(403)
 }
 expect(starts).toEqual([]);expect(cancelled).toEqual([])
})
test('unknown jobs and invalid JSON are rejected',async()=>{
 expect((await fetch(base+'/api/automation/missing')).status).toBe(404)
 expect((await fetch(base+'/api/automation',{method:'POST',body:'null'})).status).toBe(400)
 expect(starts).toEqual([])
})

test.each(['limited','uncertain'])('completed animations with %s checks remain review work in the exercise API',async(key)=>{
 const slug='barbell-bent-over-row'
 store.addReference(slug,Buffer.from('reference picture'),'reference.png')
 const run=store.createRun(slug,{automationJobId:job.id})
 store.writeRun({...run,status:'done',summary:{pass:8,fail:0,limited:0,uncertain:0,unmeasured:0,[key]:1}})

 const exercises=await(await fetch(base+'/api/exercises')).json()

 expect(exercises.find(exercise=>exercise.slug===slug).status).toBe('review')
})

test('only complete passing automation summaries show a passing exercise status',async()=>{
 const slug='barbell-bent-over-row'
 store.addReference(slug,Buffer.from('reference picture'),'reference.png')
 const run=store.createRun(slug,{automationJobId:job.id})
 const status=async()=>{
  const exercises=await(await fetch(base+'/api/exercises')).json()
  return exercises.find(exercise=>exercise.slug===slug).status
 }
 store.writeRun({...run,status:'done',summary:{pass:8,fail:0,limited:0,unmeasured:0}})
 expect(await status()).toBe('review')
 store.writeRun({...run,status:'done',summary:{pass:8,fail:0,limited:0,uncertain:0,unmeasured:0}})
 expect(await status()).toBe('passing')
})
