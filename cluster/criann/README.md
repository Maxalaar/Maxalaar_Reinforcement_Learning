> **Moved from the thesis repository (Clustering_Agent_Latent_Space), not yet run on CRIANN.**
> Paths and the installation script were updated for this project; the module `aidl/pytorch/2.2.0-cuda12.1`
> must provide Python ≥ 3.12 (requires-python in pyproject.toml). Submission scripts still point to
> `configurations/criann/<name>.py`: only `cart_pole` was migrated, the other configurations are in the thesis
> repository (`configurations/experimentation/criann/`) and need to be copied when used again.

# **CRIANN Workflow Guide**

This guide provides a streamlined workflow for connecting to CRIANN, managing Git, submitting and monitoring jobs, and setting up SSH keys for seamless access.

---

## **1. Connect to CRIANN**
To connect to the CRIANN server and navigate to your project directory:
```bash
ssh -l <login> austral.criann.fr
cd ~/Programming_Projects/Maxalaar_Reinforcement_Learning/
```

---

## **2. Git Commands**
Here are some essential Git commands for version control:

- **View unstaged changes**:
  ```bash
  git diff
  ```

- **Update local repository with remote changes**:
  ```bash
  git pull
  ```

- **Upload local commits to remote repository**:
  ```bash
  git push
  ```

- **Temporarily save uncommitted changes**:
  ```bash
  git stash
  ```

---

## **3. Job Submission and Management**

### **Submit a Job**
To submit a job using a submission script:
```bash
sbatch <submission_script>
```

The available partitions are detailed at https://services.criann.fr/services/hpc/cluster-austral/guide/#les-partitions-classes-de-soumission

### **Check Active Jobs**
View your active jobs:
```bash
squeue --me
```

### **Navigate to Job Directory**
Access the execution directory of a specific job:
```bash
cd /dlocal/run/<job_id>
```

### **Read Job Logs**
View job logs:
```bash
less ~/Programming_Projects/Maxalaar_Reinforcement_Learning/criann_logs/<file_name>
```

### **Monitor Logs in Real-Time**
Follow logs as they update:
```bash
tail -f ~/Programming_Projects/Maxalaar_Reinforcement_Learning/criann_logs/<file_name>
```

### **Connect to Compute Node**
SSH into the compute node running your job:
```bash
ssh <NODELIST(REASON)>
```

### **View Job History**
Check job history:
```bash
sacct
```

---

## **4. Cancel Jobs**

### **Cancel a Specific Job**
```bash
scancel <job_id>
```

### **Cancel All Your Jobs**
```bash
scancel --me
```

---

## **5. SSH Key Setup**

### **Generate an SSH Key**
Create a new SSH key:
```bash
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```

### **Copy Public Key to CRIANN**
Add your public key to the CRIANN server:
```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub <login>@austral.criann.fr
```

---

## **6. Environment Setup**
Install required Python packages:
```bash
./criann/python_environment/install_packages_criann.sh
```

---

## **7. Retrieving Results**

### **Sync Results from CRIANN**
Use `rsync` to copy files from the remote server to your local machine:
```bash
rsync -avz <login>@austral.criann.fr:~/Programming_Projects/Maxalaar_Reinforcement_Learning/experiments/ ./experiments/criann/
```

### **Automate Sync Every 5 Minutes**
Run the following script to sync results periodically:
```bash
while true; do rsync -avz <login>@austral.criann.fr:~/Programming_Projects/Maxalaar_Reinforcement_Learning/experiments/ ./experiments/criann/; sleep 300; done
```

---

## **8. GitHub SSH Setup**

### **Check for Existing SSH Key**
Verify if you already have an SSH key:
```bash
ls ~/.ssh/id_rsa.pub
```
If the file exists, skip to the next step. Otherwise, generate a new key.

### **Generate a New SSH Key**
```bash
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```
(Press "Enter" to save it in `~/.ssh/id_rsa`.)

### **Add SSH Key to GitHub**
1. Copy your SSH key:
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
2. Go to **GitHub → Profile → Settings → SSH and GPG keys**.
3. Click **New SSH key**, paste the key, and save.

### **Update Repository URL to Use SSH**
Change the remote URL to use SSH:
```bash
git remote set-url origin git@github.com:your-username/your-repository.git
```

### **Test SSH Connection**
Verify the setup:
```bash
ssh -T git@github.com
```
If successful, you’ll see:
```
Hi your-username! You've successfully authenticated, but GitHub does not provide shell access.
```

Now, `git pull` and other Git operations will work without requiring a password