import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="21.4 Deployment & SRE Operations - Vector DB Curriculum",
    page_icon="🚀",
    layout="wide"
)

# Title & Overview
st.title("21.4 Production Deployment: Docker, Kubernetes, Backups, Health Checks & SRE Operations")
st.caption("Chapter 21: Production Vector Database Architecture | Domain: 4. Vector Database")
st.markdown("""
Deploying a vector database in production is vastly different from deploying a stateless web API. 
Vector databases are **Stateful, Memory-Intensive, and I/O-Bound Workloads**.

If a Kubernetes pod crashes while rebuilding its HNSW graph, naive liveness probes will mark the container as unhealthy, 
triggering a catastrophic restart loop that permanently locks SQLite files. 
Furthermore, taking file-level backups while SQLite is actively writing causes corrupt, unrecoverable snapshots.

To achieve enterprise **99.99% availability and continuous uptime**, Site Reliability Engineers (SREs) enforce 
five production deployment disciplines:
1. **Hardened Docker Containerization**: Running as unprivileged non-root users with isolated volumes.
2. **Kubernetes StatefulSets with Fast NVMe Storage**: Ensuring stable network identities and persistent volume claims (PVC).
3. **12-Factor Environment Secret Hygiene**: Injecting credentials dynamically from HashiCorp Vault or AWS Secrets Manager.
4. **Consistent Automated Backups**: Running WAL checkpoints (`PRAGMA wal_checkpoint(TRUNCATE)`) before syncing snapshots to S3.
5. **Calibrated Health Probes**: Gracefully distinguishing between initial HNSW graph loading and true container deadlocks.

In this lesson, we explore production deployment manifests, build an automated backup and disaster recovery pipeline, 
examine Kubernetes health check tuning, and test an interactive Disaster Recovery & Failover Simulator.
""")

st.divider()

# Section 1: The Production Deployment Matrix
st.header("1. The Production Deployment Matrix", divider="rainbow")

col_mat1, col_mat2 = col_mat1, col_mat2 = st.columns([1.1, 0.9])

with col_mat1:
    st.markdown("""
    ### The Five Operational Tenets
    
    * **1. Docker Containerization**:
      * Use official minimal base images (`chromadb/chroma:latest`).
      * Restrict container privileges (`USER chroma`, `readOnlyRootFilesystem: true` with writable `/chroma_data` mount).
    
    * **2. Kubernetes StatefulSet Architecture**:
      * Deploy as a **StatefulSet** rather than a Deployment.
      * Mount low-latency NVMe SSDs via `PersistentVolumeClaim` (AWS `gp3` or `io2` with high IOPS).
    
    * **3. Environment Variable & Secret Hygiene**:
      * Never bake API keys into Dockerfile images.
      * Inject secrets at pod startup via Kubernetes `Secret` or AWS Secrets Manager CSI drivers.
    
    * **4. Point-In-Time Disaster Recovery**:
      * Trigger automated snapshot cron jobs: flush the write-ahead log, capture atomic snapshots of both SQLite and HNSW `.bin` files, and stream encrypted archives to multi-region S3 buckets.
    
    * **5. Calibrated Readiness Probes**:
      * Setting `initialDelaySeconds: 45` allows the database to load large HNSW graphs into RAM before receiving live query traffic.
    """)

with col_mat2:
    st.markdown("### Kubernetes StatefulSet Topology")
    k8s_dot = """
    digraph K8sTopology {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Ingress [label="Ingress Controller / Service Mesh\\nRoutes traffic based on Readiness Probe", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_statefulset {
            label="Kubernetes StatefulSet (chroma-cluster)"; style="dashed"; color="#059669";
            Pod0 [label="chroma-0 (Leader / Primary)\\nHandles Writes + Reads", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            Pod1 [label="chroma-1 (Follower Replica)\\nRead-Only Search Pool", fillcolor="#FEF3C7"];
            Pod2 [label="chroma-2 (Follower Replica)\\nRead-Only Search Pool", fillcolor="#FEF3C7"];
        }

        subgraph cluster_storage {
            label="Persistent Volume Claims (PVC)"; style="dashed"; color="#D97706";
            EBS0 [label="Fast NVMe EBS gp3\\n(10,000 IOPS /chroma_data)", fillcolor="#FDE68A"];
            S3Backup [label="Offsite S3 Bucket\\nEncrypted Nightly Snapshots", fillcolor="#E0E7FF", style="filled,bold"];
        }

        Ingress -> Pod0 [label="Write Traffic"];
        Ingress -> Pod1 [label="Search Traffic"];
        Ingress -> Pod2 [label="Search Traffic"];
        Pod0 -> EBS0 [label="Direct I/O"];
        Pod0 -> S3Backup [label="Nightly Snapshot Cron", style="dotted", color="#1E40AF"];
    }
    """
    st.graphviz_chart(k8s_dot, use_container_width=False)

st.divider()

# Section 2: Production Docker & Kubernetes Manifests
st.header("2. Production Docker & Kubernetes Manifests", divider="rainbow")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### 1. Production Docker Compose Stack")
    st.code("""version: '3.8'

services:
  chroma:
    image: chromadb/chroma:0.5.15
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"  # Expose to localhost only
    environment:
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
      - CHROMA_SERVER_AUTH_CREDENTIALS_FILE=/chroma/tokens.json
      - PERSIST_DIRECTORY=/chroma_data
      - ANONYMIZED_TELEMETRY=False
    volumes:
      - chroma_nvme_storage:/chroma_data
      - ./server_tokens.json:/chroma/tokens.json:ro
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
        reservations:
          cpus: '2.0'
          memory: 8G

volumes:
  chroma_nvme_storage:
    driver: local
""", language="yaml")

with col_code2:
    st.markdown("### 2. Kubernetes StatefulSet Snippet (Probes & PVC)")
    st.code("""apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chroma-vector-store
spec:
  serviceName: "chroma"
  replicas: 2
  template:
    spec:
      containers:
      - name: chroma
        image: chromadb/chroma:0.5.15
        resources:
          requests:
            cpu: "2000m"
            memory: "8Gi"
        # Calibrated probes preventing restart loops during index load
        readinessProbe:
          httpGet:
            path: /api/v1/heartbeat
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /api/v1/heartbeat
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 15
        volumeMounts:
        - name: data-volume
          mountPath: /chroma_data
  volumeClaimTemplates:
  - metadata:
      name: data-volume
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: "ebs-gp3-fast"
      resources:
        requests:
          storage: 100Gi
""", language="yaml")

st.divider()

# Section 3: Automated Disaster Recovery & Backup Architecture
st.header("3. Automated Disaster Recovery & Backup Architecture", divider="rainbow")

col_bk1, col_bk2 = col_bk1, col_bk2 = st.columns([1, 1])

with col_bk1:
    st.markdown("""
    ### Why Naive File Copying Fails
    * ChromaDB persists metadata in `chroma.sqlite3` with an active Write-Ahead Log (`chroma.sqlite3-wal`).
    * If you run `cp -r /chroma_data /backups` while a client is writing, the SQLite database file and WAL file are copied at different timestamps.
    * The resulting backup is **corrupt and unopenable**!
    
    ### The Three-Step Atomic Backup Protocol
    1. **Step 1: Checkpoint SQLite**: Execute `PRAGMA wal_checkpoint(TRUNCATE)` to flush all pending transactions from the WAL file into the primary database file.
    2. **Step 2: Filesystem Snapshot / Tar**: Create a compressed tar archive of the storage folder while in a quiescent state or take an EBS volume snapshot.
    3. **Step 3: Offsite Encrypted Upload**: Stream the archive to an encrypted S3 bucket (`aws s3 cp backup.tar.gz s3://corp-backups/vector-db/ --sse aws:kms`).
    """)

with col_bk2:
    st.markdown("### Automated Python Backup Script")
    st.code("""import sqlite3
import tarfile
import os
import datetime
import boto3

def execute_atomic_chroma_backup(data_dir: str = "/chroma_data", s3_bucket: str = "corp-vector-backups"):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_name = f"/tmp/chroma_backup_{timestamp}.tar.gz"
    sqlite_path = os.path.join(data_dir, "chroma.sqlite3")

    # 1. Atomic SQLite WAL Checkpoint
    print("Flushing SQLite WAL checkpoint...")
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

    # 2. Package database and HNSW graph binaries
    print(f"Creating tar archive {archive_name}...")
    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(data_dir, arcname="chroma_data")

    # 3. Stream to S3 with KMS Encryption
    print("Streaming backup to AWS S3...")
    s3_client = boto3.client("s3")
    s3_key = f"backups/chroma/{timestamp}/backup.tar.gz"
    s3_client.upload_file(
        archive_name,
        s3_bucket,
        s3_key,
        ExtraArgs={"ServerSideEncryption": "aws:kms"}
    )
    os.remove(archive_name)
    print(f"Backup complete! Stored at s3://{s3_bucket}/{s3_key}")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Production Disaster Recovery Simulator
st.header("4. Interactive Workbench: Production Disaster Recovery Simulator", divider="rainbow")

st.markdown("""
Simulate production failure events: an unannounced node crash, volume corruption, or failover. 
Test how the backup and health probe mechanisms restore system health to 100%.
""")

col_sim_event, col_sim_ctrl = col_sim_event, col_sim_ctrl = st.columns([1, 1.2])

with col_sim_event:
    st.subheader("1. Inject Production Incident")
    incident_type = st.selectbox(
        "Choose Failure Scenario:",
        [
            "Scenario A: Primary Node Crash & Automatic Kubernetes Pod Reschedule",
            "Scenario B: Disk Volume Corruption & S3 Snapshot Restoration",
            "Scenario C: Slow Cold-Start HNSW Load & Readiness Probe Traffic Gating"
        ]
    )

with col_sim_ctrl:
    st.subheader("2. Cluster Health Telemetry")
    if "Scenario A" in incident_type:
        s_status = "🟡 Pod Evicted -> Rescheduling on Node B"
        s_rto = "45 Seconds (RTO)"
        s_rpo = "0 Seconds (Persistent Volume Claim Retained)"
        s_log = "[EVENT] Kubelet: Pod chroma-0 failed liveness. Rescheduling StatefulSet pod with existing PVC volume claim. Re-attached NVMe disk in 12s."
    elif "Scenario B" in incident_type:
        s_status = "🔴 Volume Unrecoverable -> S3 Snapshot Restore Triggered"
        s_rto = "8 Minutes (RTO)"
        s_rpo = "2 Hours (Last Nightly S3 Snapshot)"
        s_log = "[EVENT] SRE Alert: Corrupted NVMe sector detected. Rebuilt new EBS volume. Restored from s3://corp-backups/20241013.tar.gz. 1.2M vectors restored."
    else:  # Scenario C
        s_status = "🟢 Readiness Probe Protecting Live Traffic"
        s_rto = "0 Seconds (Zero Dropped Packets)"
        s_rpo = "0 Seconds"
        s_log = "[EVENT] Pod chroma-1 starting up. HNSW graph loading (18s). Readiness probe HTTP 503: traffic routed exclusively to chroma-0. Node became ready at 22s."

    st.metric(label="Cluster Health Status", value=s_status)
    m_rto, m_rpo = st.columns(2)
    m_rto.metric(label="Recovery Time Objective (RTO)", value=s_rto)
    m_rpo.metric(label="Recovery Point Objective (RPO)", value=s_rpo)

st.divider()

st.subheader("Incident Timeline & System Audit Log")
st.code(s_log, language="text")

if "Scenario C" in incident_type:
    st.success("✅ SRE Excellence: Because `readinessProbe` had an appropriate `initialDelaySeconds`, zero user queries were dropped during cold index startup!")
elif "Scenario A" in incident_type:
    st.info("💡 High Availability Verified: Kubernetes StatefulSets re-attach existing EBS volumes without data loss.")
else:
    st.warning("⚠️ Disaster Recovery Verified: S3 snapshots restored the database, meeting the corporate Recovery Time Objective (RTO).")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why must vector databases in Kubernetes be deployed as StatefulSets rather than standard Deployments?"):
    st.markdown("""
    **Answer**:
    1. **Stable Network Identifiers**: In a distributed cluster (e.g. Qdrant or Milvus), nodes must locate each other via predictable DNS hostnames (`chroma-0.chroma.svc.cluster.local`, `chroma-1...`). Deployments assign random pod hashes that break cluster consensus.
    2. **Dedicated Persistent Volume Claims (PVC)**: StatefulSets use `volumeClaimTemplates` to ensure that each pod is bound to its own dedicated NVMe volume. If `chroma-0` crashes, Kubernetes re-spins it on another host and **re-attaches the exact same NVMe disk**, eliminating lengthy data re-syncs.
    3. **Ordered Graceful Shutdown**: StatefulSets terminate pods in strict reverse order (`chroma-2` -> `chroma-1` -> `chroma-0`), allowing SQLite write-ahead logs to flush cleanly.
    """)

with st.expander("Q2: What is the difference between a Liveness Probe and a Readiness Probe for a vector database?"):
    st.markdown("""
    **Answer**:
    * **Readiness Probe (`/api/v1/heartbeat`)**:
      * Determines whether the container is **ready to accept user traffic**.
      * When a vector database starts up, it must read binary files from disk into RAM to rebuild its HNSW graph (which can take 15–45 seconds).
      * While loading, the readiness probe fails, instructing the load balancer **NOT to route search queries to this pod**.
    * **Liveness Probe**:
      * Determines whether the container is **still alive or deadlocked**.
      * If a liveness probe fails, Kubernetes forcefully kills and restarts the container (`SIGKILL`).
      * **Critical Trap**: If the liveness probe fires while the database is still loading its graph, Kubernetes kills the pod, creating an infinite restart crash loop! Liveness probes must have a generous `initialDelaySeconds` (e.g. 60–90s).
    """)

with st.expander("Q3: How do you perform zero-downtime rolling upgrades of a vector database cluster?"):
    st.markdown("""
    **Answer**:
    1. The cluster must have at least 2 read replicas (`chroma-0` and `chroma-1`).
    2. Configure the StatefulSet with `updateStrategy: type: RollingUpdate, partition: 1`.
    3. Upgrade `chroma-1` first. Traffic shifts entirely to `chroma-0`.
    4. Wait for `chroma-1` to pass its readiness probe and re-join the search pool.
    5. Upgrade `chroma-0`. Traffic shifts to `chroma-1`.
    6. Once both pods are healthy, the rolling upgrade completes with **zero failed user queries**.
    """)

with st.expander("Q4: Why should vector database volumes use fast NVMe SSD storage (AWS gp3 / io2) rather than standard HDD or EFS network shares?"):
    st.markdown("""
    **Answer**:
    * Vector database storage involves **high-frequency random 4KB read/write operations** across SQLite B-Trees and HNSW binary segments.
    * Network-attached filesystems like AWS EFS have high latency (10ms–30ms per read), completely destroying search speed.
    * Standard spinning HDDs cannot handle SQLite WAL fsync operations, capping ingestion throughput at under 50 vectors/sec.
    * Provisioned NVMe SSDs (gp3 with 10,000+ IOPS) provide sub-millisecond disk access, enabling 4,000+ VPS ingestion throughput.
    """)

with st.expander("Q5: What are RTO and RPO, and how are they defined for production vector databases?"):
    st.markdown("""
    **Answer**:
    * **Recovery Time Objective (RTO)**: The maximum acceptable duration of system downtime after a failure. (e.g., *"The vector database must be restored and serving search queries within 15 minutes of a region failure"*).
    * **Recovery Point Objective (RPO)**: The maximum acceptable data loss measured in time. (e.g., *"We cannot lose more than 1 hour of newly ingested documents"*).
    * **Architecture**: To achieve RPO < 1 hour, take automated hourly incremental snapshots to S3. To achieve RTO < 5 minutes, deploy cross-zone read replicas with automated DNS failover.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Aggressive Liveness Probes on Slow Cold Starts**  
    Setting `initialDelaySeconds: 5` causes Kubernetes to kill the pod while it is legitimately loading its HNSW index into RAM, creating an infinite crash loop. Set to 60+ seconds.
    """)
    st.warning("""
    **Trap 2: Backing Up SQLite Without WAL Checkpoint**  
    Copying `chroma.sqlite3` while transactions are pending in the `-wal` file creates corrupt, unbootable backups. Always execute `PRAGMA wal_checkpoint(TRUNCATE)` first.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Running Containers as the Root User (`root`)**  
    Running Chroma as root exposes the underlying host kernel to container breakout vulnerabilities. Always configure non-root security contexts (`runAsUser: 10001`).
    """)
    st.warning("""
    **Trap 4: Missing Resource Limits (OOM Killer)**  
    Deploying without Kubernetes memory limits allows the vector database to consume all host node RAM, triggering the Linux kernel OOM Killer to kill adjacent system pods. Set explicit limits.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: Why use StatefulSets for vector DBs?"):
        st.markdown("""
        **Stable Storage & Identity**:  
        Preserves dedicated NVMe volume attachments and predictable network DNS across pod restarts.
        """)
    with st.expander("🗂️ Flashcard: What does Readiness Probe do?"):
        st.markdown("""
        **Traffic Gate**:  
        Blocks incoming search traffic until the pod finishes loading its HNSW graph into RAM.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How to take atomic SQLite backups?"):
        st.markdown("""
        **WAL Flush**:  
        Execute `PRAGMA wal_checkpoint(TRUNCATE)` before archiving database files to S3.
        """)
    with st.expander("🗂️ Flashcard: What are RTO and RPO?"):
        st.markdown("""
        **DR Metrics**:  
        RTO: Maximum allowable downtime.  
        RPO: Maximum allowable data loss window.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the production deployment matrix: Docker, Kubernetes StatefulSets, and fast NVMe storage.", value=True)
chk2 = st.checkbox("I know why Kubernetes StatefulSets are required to bind persistent volume claims to vector nodes.", value=True)
chk3 = st.checkbox("I understand how to calibrate Liveness and Readiness probes to prevent cold-start restart loops.", value=True)
chk4 = st.checkbox("I can implement an atomic SQLite WAL checkpoint and encrypted S3 backup automation pipeline.", value=True)
chk5 = st.checkbox("I know how to execute zero-downtime rolling upgrades across vector database read replicas.", value=True)
chk6 = st.checkbox("I can design, deploy, and operate an enterprise-grade 99.99% available vector search system in production.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completed Chapter 21 and 100% of the entire Domain 5: Vector Database Curriculum (All 75 Lessons)!")
