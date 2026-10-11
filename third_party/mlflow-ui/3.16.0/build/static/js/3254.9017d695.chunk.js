"use strict";(globalThis.webpackChunk_mlflow_mlflow=globalThis.webpackChunk_mlflow_mlflow||[]).push([[3254],{8691(e,r,t){t.d(r,{f:()=>d});var a=t(15032),o=t(10587),n=t(51477),l=t(71685),i=t(86856),s=t(63968);const d=({datasetName:e,datasetDigest:r,runId:t})=>{const{theme:d}=(0,o.u)(),[u,c]=(0,n.useState)(!1),{onDatasetClicked:v}=(0,l.s7)(),{handleError:m}=(0,i.tF)();return(0,s.FD)(o.B,{type:"link",icon:u?(0,s.Y)(o.ah,{size:"small",css:(0,a.AH)({marginRight:d.spacing.sm},"")}):(0,s.Y)(o.hm,{}),componentId:"mlflow.logged_model.dataset",onClick:()=>((e,r,t)=>{t&&(c(!0),null===v||void 0===v||v({datasetName:e,datasetDigest:r,runId:t}).catch(e=>{m(e)}).finally(()=>c(!1)))})(e,r,t),children:[e," (#",r,")"]},[e,r].join("."))}},42213(e,r,t){t.d(r,{J:()=>h});var a=t(15032),o=t(77753),n=t(10587),l=t(67162),i=t(58213),s=t(67593),d=t(51605),u=t(23146),c=t(51477),v=t(39521),m=t(63968);var p={name:"1wcfv52",styles:"margin-right:0"},g={name:"1a7v7i3",styles:"margin-right:0;&>div{padding-right:0;}"};const h=({loggedModel:e,displayDetails:r,className:t})=>{var h,f,E,y;const[D]=(0,v.ok)(),k=(0,c.useMemo)(()=>{var r,t,a;return null!==(r=null===e||void 0===e||null===(t=e.info)||void 0===t||null===(a=t.tags)||void 0===a?void 0:a.reduce((e,r)=>r.key?(e[r.key]=r,e):e,{}))&&void 0!==r?r:{}},[null===e||void 0===e||null===(h=e.info)||void 0===h?void 0:h.tags]),_=null===k||void 0===k||null===(f=k[s.xd])||void 0===f?void 0:f.value,I=null===k||void 0===k||null===(E=k[i.A.gitCommitTag])||void 0===E?void 0:E.value,M=(0,c.useMemo)(()=>{try{return i.A.renderSource(k,D.toString(),void 0,_)}catch(e){return}},[k,D,_]),A=null===(y=k[i.A.sourceTypeTag])||void 0===y?void 0:y.value,{theme:w}=(0,n.u)();return M?(0,m.FD)("div",{css:(0,a.AH)({display:"flex",alignItems:"center",gap:w.spacing.sm,paddingTop:w.spacing.sm,paddingBottom:w.spacing.sm,flexWrap:r?"wrap":void 0},""),className:t,children:[A&&(0,m.Y)(u.m,{sourceType:A,css:(0,a.AH)({color:w.colors.actionPrimaryBackgroundDefault},"")}),M," ",r&&_&&(0,m.Y)(n.s,{componentId:"mlflow.logged_model.details.source.branch_tooltip",content:_,children:(0,m.Y)(o.Tag,{componentId:"mlflow.logged_model.details.source.branch",css:p,children:(0,m.FD)("div",{css:(0,a.AH)({display:"flex",gap:w.spacing.xs,whiteSpace:"nowrap"},""),children:[(0,m.Y)(n.bW,{})," ",_]})})}),r&&I&&(0,m.FD)(l.j.Root,{componentId:"mlflow.logged_model.details.source.commit_hash_popover",children:[(0,m.Y)(l.j.Trigger,{asChild:!0,children:(0,m.Y)(o.Tag,{componentId:"mlflow.logged_model.details.source.commit_hash",css:g,children:(0,m.FD)("div",{css:(0,a.AH)({display:"flex",gap:w.spacing.xs,whiteSpace:"nowrap",alignContent:"center"},""),children:[(0,m.Y)(n.ef,{}),I.slice(0,7)]})})}),(0,m.FD)(l.j.Content,{align:"start",children:[(0,m.Y)(l.j.Arrow,{}),(0,m.FD)("div",{css:(0,a.AH)({display:"flex",gap:w.spacing.xs,alignItems:"center"},""),children:[I,(0,m.Y)(d.i,{showLabel:!1,size:"small",type:"tertiary",copyText:I,icon:(0,m.Y)(n.cV,{})})]})]})]})]}):(0,m.Y)(n.T.Hint,{children:"\u2014"})}},78486(e,r,t){t.d(r,{a:()=>u});var a=t(15032),o=t(77753),n=t(10587),l=t(5460),i=t(20285),s=t(63968);const d=({status:e})=>e===i.Fq.LOGGED_MODEL_READY?(0,s.Y)(n.aA,{color:"success"}):e===i.Fq.LOGGED_MODEL_UPLOAD_FAILED?(0,s.Y)(n.ic,{color:"danger"}):e===i.Fq.LOGGED_MODEL_PENDING?(0,s.Y)(n.ae,{color:"warning"}):null,u=({data:e})=>{var r,t;const{theme:u}=(0,n.u)(),c=null!==(r=null===(t=e.info)||void 0===t?void 0:t.status)&&void 0!==r?r:i.Fq.LOGGED_MODEL_STATUS_UNSPECIFIED;return c===i.Fq.LOGGED_MODEL_STATUS_UNSPECIFIED?null:(0,s.FD)(o.Tag,{componentId:"mlflow.logged_model.status",css:(0,a.AH)({backgroundColor:c===i.Fq.LOGGED_MODEL_READY?u.isDarkMode?u.colors.green800:u.colors.green100:c===i.Fq.LOGGED_MODEL_UPLOAD_FAILED?u.isDarkMode?u.colors.red800:u.colors.red100:c===i.Fq.LOGGED_MODEL_PENDING?u.isDarkMode?u.colors.yellow800:u.colors.yellow100:void 0},""),children:[c&&(0,s.Y)(d,{status:c})," ",(0,s.Y)(n.T.Text,{css:(0,a.AH)({marginLeft:u.spacing.sm},""),children:c===i.Fq.LOGGED_MODEL_READY?(0,s.Y)(n.T.Text,{color:"success",children:(0,s.Y)(l.A,{id:"Rs+SVS",defaultMessage:"Ready"})}):c===i.Fq.LOGGED_MODEL_UPLOAD_FAILED?(0,s.Y)(n.T.Text,{color:"error",children:(0,s.Y)(l.A,{id:"e6reUn",defaultMessage:"Failed"})}):c===i.Fq.LOGGED_MODEL_PENDING?(0,s.Y)(n.T.Text,{color:"warning",children:(0,s.Y)(l.A,{id:"jo4LfR",defaultMessage:"Pending"})}):c})]})}},21379(e,r,t){t.d(r,{P:()=>n});t(51477);var a=t(23984),o=t(63968);const n=({value:e})=>{const r=new Date(Number(e));return isNaN(r)?null:(0,o.Y)(a.f,{date:r})}},71685(e,r,t){t.d(r,{Xs:()=>p,s7:()=>g});var a=t(51477),o=t(35297),n=t(71770),l=t(13288),i=t(29942),s=t(68073),d=t(88843),u=t(5460),c=t(63968);class v extends s.ZR{constructor(...e){super(...e),this.errorLogType=s.ZQ.UnexpectedSystemStateError,this.errorName=s.UW.DatasetRunNotFoundError,this.isUserError=!0,this.displayMessage=(0,c.Y)(u.A,{id:"vwDBPr",defaultMessage:"The run containing the dataset could not be found."})}}const m=(0,a.createContext)({onDatasetClicked:()=>Promise.resolve()}),p=({children:e})=>{const[r,t]=(0,a.useState)(!1),[s,u]=(0,a.useState)(),[p]=(0,n.T)(),g=(0,a.useRef)(null),h=(0,a.useCallback)(async e=>new Promise((r,a)=>{var o;return null===(o=g.current)||void 0===o||o.call(g),p({onError:a,onCompleted(o){var n,s,c,m,p,h,f,E,y,D,k;if(null!==(n=o.mlflowGetRun)&&void 0!==n&&n.apiError){const e=o.mlflowGetRun.apiError.code===d.tG.RESOURCE_DOES_NOT_EXIST?new v:o.mlflowGetRun.apiError;return void a(e)}const _=(0,l.u)(null===(s=o.mlflowGetRun)||void 0===s||null===(c=s.run)||void 0===c||null===(m=c.inputs)||void 0===m?void 0:m.datasetInputs);if(!_||null===(p=o.mlflowGetRun)||void 0===p||null===(h=p.run)||void 0===h||!h.info)return void r();const I=null===_||void 0===_?void 0:_.find(r=>{var t;return(null===(t=r.dataset)||void 0===t?void 0:t.digest)===e.datasetDigest&&r.dataset.name===e.datasetName});if(!I)return void r();const{info:M,data:A}=o.mlflowGetRun.run,w=(0,i.keyBy)(null!==(f=null===A||void 0===A||null===(E=A.tags)||void 0===E?void 0:E.filter(e=>e.key&&e.value))&&void 0!==f?f:[],"key");t(!0),u({datasetWithTags:{dataset:I.dataset,tags:I.tags},runData:{datasets:_,runUuid:null!==(y=M.runUuid)&&void 0!==y?y:"",experimentId:null!==(D=M.experimentId)&&void 0!==D?D:"",runName:null!==(k=M.runName)&&void 0!==k?k:"",tags:w}}),r(),g.current=null},variables:{data:{runId:e.runId}}})}),[p]),f=(0,a.useMemo)(()=>({onDatasetClicked:h}),[h]);return(0,c.FD)(m.Provider,{value:f,children:[e,s&&(0,c.Y)(o.O,{isOpen:r,selectedDatasetWithRun:s,setIsOpen:t,setSelectedDatasetWithRun:u})]})},g=()=>(0,a.useContext)(m)},71770(e,r,t){t.d(r,{T:()=>l,t:()=>n});var a=t(70270);const o=a.J1`
  query GetRun($data: MlflowGetRunInput!) @component(name: "MLflow.ExperimentRunTracking") {
    mlflowGetRun(input: $data) {
      apiError {
        helpUrl
        code
        message
      }
      run {
        info {
          runName
          status
          runUuid
          experimentId
          artifactUri
          endTime
          lifecycleStage
          startTime
          userId
        }
        experiment {
          experimentId
          name
          tags {
            key
            value
          }
          artifactLocation
          lifecycleStage
          lastUpdateTime
        }
        modelVersions {
          status
          version
          name
          source
        }
        data {
          metrics {
            key
            value
            step
            timestamp
          }
          params {
            key
            value
          }
          tags {
            key
            value
          }
        }
        inputs {
          datasetInputs {
            dataset {
              digest
              name
              profile
              schema
              source
              sourceType
            }
            tags {
              key
              value
            }
          }
          modelInputs {
            modelId
          }
        }
        outputs {
          modelOutputs {
            modelId
            step
          }
        }
      }
    }
  }
`,n=({runUuid:e,disabled:r=!1})=>{var t,n;const{data:l,loading:i,error:s,refetch:d}=(0,a.IT)(o,{variables:{data:{runId:e}},skip:r});return{loading:i,data:null===l||void 0===l||null===(t=l.mlflowGetRun)||void 0===t?void 0:t.run,refetchRun:d,apolloError:s,apiError:null===l||void 0===l||null===(n=l.mlflowGetRun)||void 0===n?void 0:n.apiError}},l=()=>(0,a._l)(o)},13288(e,r,t){t.d(r,{u:()=>p,g:()=>g});var a=t(29942),o=t(51477),n=t(34665),l=t(4981),i=t(10413),s=t(58213);var d=t(71770),u=t(89511),c=t(45766);const v=({queryResult:e,runUuid:r})=>{const{registeredModels:t}=(0,n.d4)(({entities:e})=>({registeredModels:e.modelVersionsByRunUuid[r]}));if((0,u._O)()){const r=[];var a,o;if(null!==e&&void 0!==e&&e.data&&"modelVersions"in e.data)null===(a=e.data)||void 0===a||null===(o=a.modelVersions)||void 0===o||o.forEach(e=>{r.push({displayedName:e.name,version:e.version,link:e.name&&e.version?c.fM.getModelVersionPageRoute(e.name,e.version):"",status:e.status,source:e.source})});return r}return t?t.map(e=>{const r=e.name,t=c.fM.getModelVersionPageRoute(r,e.version);return{displayedName:e.name,version:e.version,link:t,status:e.status,source:e.source}}):[]},m=e=>(0,a.keyBy)(e,"key"),p=e=>null===e||void 0===e?void 0:e.map(e=>{var r,t,o,n,l,i,s,d,u,c,v,m,p,g;return{dataset:{digest:null!==(r=null===(t=e.dataset)||void 0===t?void 0:t.digest)&&void 0!==r?r:"",name:null!==(o=null===(n=e.dataset)||void 0===n?void 0:n.name)&&void 0!==o?o:"",profile:null!==(l=null===(i=e.dataset)||void 0===i?void 0:i.profile)&&void 0!==l?l:"",schema:null!==(s=null===(d=e.dataset)||void 0===d?void 0:d.schema)&&void 0!==s?s:"",source:null!==(u=null===(c=e.dataset)||void 0===c?void 0:c.source)&&void 0!==u?u:"",sourceType:null!==(v=null===(m=e.dataset)||void 0===m?void 0:m.sourceType)&&void 0!==v?v:""},tags:null!==(p=null===(g=e.tags)||void 0===g?void 0:g.map(e=>{var r,t;return{key:null!==(r=e.key)&&void 0!==r?r:"",value:null!==(t=e.value)&&void 0!==t?t:""}}).filter(e=>!(0,a.isEmpty)(e.key)))&&void 0!==p?p:[]}}),g=({runUuid:e,experimentId:r})=>{var t,c,g,h,f,E;const y=(0,u.wD)(),D=(0,n.wA)(),k=!0;if(y){var _,I,M,A,w,R;const r=(()=>(0,d.t)({runUuid:e}))();(0,o.useEffect)(()=>{(0,u._O)()||D((0,i.hY)({run_id:e}))},[D,e,k]);const{latestMetrics:t,tags:n,params:l,datasets:s}=(0,o.useMemo)(()=>{var e,t,o,n,l,i,s,d,u,c,v,g;return{latestMetrics:(0,a.pickBy)(m((g=null!==(e=null===(t=r.data)||void 0===t||null===(o=t.data)||void 0===o?void 0:o.metrics)&&void 0!==e?e:[],g.filter(({key:e,value:r,step:t,timestamp:a})=>null!==e&&null!==r&&null!==t&&null!==a).map(({key:e,value:r,step:t,timestamp:a})=>({key:e,value:r,step:Number(t),timestamp:Number(a)})))),e=>e.key.trim().length>0),tags:(0,a.pickBy)(m(null!==(n=null===(l=r.data)||void 0===l||null===(i=l.data)||void 0===i?void 0:i.tags)&&void 0!==n?n:[]),e=>e.key.trim().length>0),params:(0,a.pickBy)(m(null!==(s=null===(d=r.data)||void 0===d||null===(u=d.data)||void 0===u?void 0:u.params)&&void 0!==s?s:[]),e=>e.key.trim().length>0),datasets:p(null===(c=r.data)||void 0===c||null===(v=c.inputs)||void 0===v?void 0:v.datasetInputs)}},[r.data]),c=v({runUuid:e,queryResult:r});return{runInfo:null!==(_=null===(I=r.data)||void 0===I?void 0:I.info)&&void 0!==_?_:void 0,experiment:null!==(M=null===(A=r.data)||void 0===A?void 0:A.experiment)&&void 0!==M?M:void 0,loading:r.loading,error:r.apolloError,apiError:r.apiError,refetchRun:r.refetchRun,runInputs:null===(w=r.data)||void 0===w?void 0:w.inputs,runOutputs:null===(R=r.data)||void 0===R?void 0:R.outputs,registeredModelVersionSummaries:c,datasets:s,latestMetrics:t,tags:n,params:l}}const T=((e,r,t=!0)=>{const[d,u]=(0,o.useState)(""),[c,v]=(0,o.useState)(""),m=(0,n.wA)(),{runInfo:p,tags:g,latestMetrics:h,experiment:f,params:E,datasets:y}=(0,n.d4)(t=>({runInfo:t.entities.runInfosByUuid[e],tags:(0,a.pickBy)(t.entities.tagsByRunUuid[e],e=>e.key.trim().length>0),latestMetrics:(0,a.pickBy)(t.entities.latestMetricsByRunUuid[e],e=>e.key.trim().length>0),params:(0,a.pickBy)(t.entities.paramsByRunUuid[e],e=>e.key.trim().length>0),experiment:t.entities.experimentsById[r],datasets:t.entities.runDatasetsByUuid[e]})),D=(0,o.useCallback)(()=>{const r=(0,l.aO)(e);return u(r.meta.id),m(r)},[m,e]),k=(0,o.useCallback)(()=>{const e=(0,l.yc)(r);return v(e.meta.id),m(e)},[m,r]),_=(0,o.useCallback)(()=>{t&&m((0,i.hY)({run_id:e}))},[m,e,t]);(0,o.useEffect)(()=>{p||D().catch(e=>s.A.logErrorAndNotifyUser(e)),_()},[p,D,_]),(0,o.useEffect)(()=>{f||k().catch(e=>s.A.logErrorAndNotifyUser(e))},[f,k]);const{loading:I,error:M}=(0,n.d4)(e=>{var r,t,a,o;return{loading:!d||Boolean(null===(r=e.apis)||void 0===r||null===(t=r[d])||void 0===t?void 0:t.active),error:null===(a=e.apis)||void 0===a||null===(o=a[d])||void 0===o?void 0:o.error}}),{loading:A,error:w}=(0,n.d4)(e=>{var r,t,a,o;return{loading:!d||Boolean(null===(r=e.apis)||void 0===r||null===(t=r[c])||void 0===t?void 0:t.active),error:null===(a=e.apis)||void 0===a||null===(o=a[c])||void 0===o?void 0:o.error}});return{loading:I||A,data:{runInfo:p,tags:g,params:E,latestMetrics:h,experiment:f,datasets:y},refetchRun:D,errors:{runFetchError:M,experimentFetchError:w}}})(e,r,k),U=T.errors.runFetchError||T.errors.experimentFetchError,x=v({runUuid:e});return{runInfo:null===(t=T.data)||void 0===t?void 0:t.runInfo,latestMetrics:null===(c=T.data)||void 0===c?void 0:c.latestMetrics,tags:null===(g=T.data)||void 0===g?void 0:g.tags,experiment:null===(h=T.data)||void 0===h?void 0:h.experiment,params:null===(f=T.data)||void 0===f?void 0:f.params,datasets:null===(E=T.data)||void 0===E?void 0:E.datasets,loading:T.loading,error:U,runFetchError:T.errors.runFetchError,experimentFetchError:T.errors.experimentFetchError,refetchRun:T.refetchRun,registeredModelVersionSummaries:x}}},23984(e,r,t){t.d(r,{Z:()=>a.Z,f:()=>a.f});var a=t(994)},86856(e,r,t){t.d(r,{Au:()=>i,tF:()=>s});var a=t(51477),o=t(68073),n=t(63968);const l=(0,a.createContext)({currentUserActionError:null,handleError:()=>{},handlePromise:()=>{},clearUserActionError:()=>{}}),i=({children:e,errorFilter:r})=>{const[t,i]=(0,a.useState)(null),s=(0,a.useCallback)((e,t)=>{if(null===r||void 0===r||!r(e)){const r=(0,o.a$)(e);i(r),t&&t(r)}},[i,r]),d=(0,a.useCallback)(e=>{e.catch(e=>{s(e)})},[s]),u=(0,a.useCallback)(()=>{i(null)},[i]);return(0,n.Y)(l.Provider,{value:(0,a.useMemo)(()=>({currentUserActionError:t,handleError:s,handlePromise:d,clearUserActionError:u}),[u,t,s,d]),children:e})},s=()=>{const{currentUserActionError:e,handleError:r,handlePromise:t,clearUserActionError:o}=(0,a.useContext)(l),n=(0,a.useCallback)((e,t,a)=>{r(t,a)},[r]);return(0,a.useMemo)(()=>({currentUserActionError:e,handleError:r,handleErrorWithEvent:n,handlePromise:t,clearUserActionError:o}),[o,r,t,e,n])}}}]);