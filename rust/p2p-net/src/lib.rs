use libp2p::{
    gossipsub, identify, kad, mdns, noise, ping,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, PeerId, Swarm,
};
use std::time::Duration;
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

// the messages Lattice nodes exchange 
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LatticeMessage {
    // a node shares its computed gradient for this training step
    GradientShare{
        node_id: String,
        step: u64,
        gradient_bytes: Vec<u8>,
    },
    // a node proposes a new gradient block
    BlockProposal {
        block_bytes: Vec<u8>,
    },
    // a node accepts or rejects a proposed block
    BlockVote {
        block_hash: [u8; 32],
        accept: bool,
        voter_id: String,
    },
}

// combined network behaviour
#[derive(NetworkBehaviour)]
pub struct LatticeBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub mdns: mdns::tokio::Behaviour,
    pub ping: ping::Behaviour,
    pub identify: identify::Behaviour
}

pub const TOPIC_GRADIENTS: &str = "lattice/gradients/v1";
pub const TOPIC_BLOCK: &str = "lattice/blocks/v1";

pub async fn build_swarm() -> anyhow::Result<Swarm<LatticeBehaviour>> {
    let swarm = libp2p::SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let peer_id = PeerId::from(key.public());

            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub::Config::default(),
            )?;

            let kademlia = kad::Behaviour::new(
                peer_id,
                kad::store::MemoryStore::new(peer_id),
            );

            let mdns = mdns::tokio::Behaviour::new(
                mdns::Config::default(),
                peer_id,
            )?;

            Ok(LatticeBehaviour {
                gossipsub,
                kademlia,
                mdns,
                ping: ping::Behaviour::default(),
                identify: identify::Behaviour::new(
                    identify::Config::new("/lattice/1.0.0".to_string(), key.public())
                ),
            })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    Ok(swarm)
}

pub async fn run_node(
    mut swarm: Swarm<LatticeBehaviour>,
    listen_addr: &str,
    // channel to send received messages to the training loop
    incoming_tx: mpsc::Sender<LatticeMessage>,
    // channel to receive messages to publish from the training loop
    mut outgoing_rx: mpsc::Receiver<(String, LatticeMessage)>,
) -> anyhow::Result<()> {
    use libp2p::gossipsub::TopicHash;

    let grad_topic = gossipsub::IdentTopic::new(TOPIC_GRADIENTS);
    let block_topic = gossipsub::IdentTopic::new(TOPIC_BLOCK);
    swarm.behaviour_mut().gossipsub.subscribe(&grad_topic)?;
    swarm.behaviour_mut().gossipsub.subscribe(&block_topic)?;

    swarm.listen_on(listen_addr.parse()?)?;

    loop{
        tokio::select! {
            // handle outgoing: training loop wants to publish something
            Some((topic_name, msg)) = outgoing_rx.recv() => {
                let bytes = bincode::serialize(&msg)?;
                let topic = gossipsub::IdentTopic::new(topic_name);
                if let Err(e) = swarm.behaviour_mut().gossipsub.publish(topic, bytes) {
                    tracing::warn!("publish failed {:?}", e);
                }
            }

            //handle incoming network SwarmEvent
            event = swarm.next() => {
                match event {
                    Some(SwarmEvent::Behaviour(LatticeBehaviourEvent::Gossipsub(
                        gossipsub::Event::Message {message, .. }
                    ))) => {
                        if let Ok(msg) = bincode::deserialize::<LatticeMessage>(&message.data) {
                            let _ = incoming_tx.send(msg).await;
                        }
                    }

                    Some(SwarmEvent::Behaviour(LatticeBehaviourEvent::Mdns(
                        mdns::Event::Discovered(peers)
                    ))) => {
                        for (peer_id, addr) in peers {
                            tracing::info!("discovered peer: {peer_id} at {addr}");
                            swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                        }
                    }

                    Some(SwarmEvent::NewListenAddr {address, ..}) => {
                        tracing::info!("listening on {address}");
                    }
                    _ => {}
                }
            }
        }
    }
}
